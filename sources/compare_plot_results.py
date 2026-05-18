"""
compare_plot_results.py
=======================
Treina AGCRN como baseline, carrega o GraphWaveNet treinado e compara
as metricas de ambos os modelos sobre o conjunto de teste.

Execucao:
  python baselines/compare_plot_results.py

Ordem de execucao:
  1. AGCRN  — treinado do zero em processo filho isolado (contexto CUDA limpo)
  2. GWN    — carregado do checkpoint salvo por train.py

O AGCRN e executado via multiprocessing (spawn) para garantir contexto
CUDA limpo, evitando fragmentacao de VRAM residual do processo principal.
Resultados transferidos por arquivo .npz temporario — sem tensores PyTorch
entre processos.
"""

import os
import sys
import gc
import logging
import tempfile
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

_here       = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.normpath(os.path.join(_here, "..")),
    os.path.normpath(os.path.join(_here, "..", "sources")),
]
SRC = None
for _c in _candidates:
    if os.path.isfile(os.path.join(_c, "config.py")):
        SRC = _c
        break
if SRC is None:
    raise RuntimeError(
        "Nao foi possivel localizar sources/config.py.\n"
        f"  Caminhos tentados: {_candidates}"
    )
if SRC not in sys.path:
    sys.path.insert(0, SRC)

_stream_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8",
                closefd=False, buffering=1)
)
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler("baselines.log", encoding="utf-8"),
        _stream_handler,
    ],
)
log = logging.getLogger(__name__)

_SEP_THIN = "-" * 60
_SEP_WIDE = "=" * 90


# ─────────────────────────────────────────────────────────────────────────────
# Mascara de nos ativos — criterio em escala real
# ─────────────────────────────────────────────────────────────────────────────

def build_active_mask(train_mean: pd.DataFrame,
                      node_order: list,
                      min_mean_pass: float = 1.0) -> np.ndarray:
    """
    Identifica nos com dado real suficiente usando escala original (passageiros).

    Criterio: demanda media no treino >= min_mean_pass passageiros/janela.
    Usa train_mean.csv (escala real), nao variancia do tensor z-score.
    Criterio interpretavel e consistente com evaluate.py.
    """
    means      = train_mean["loading"].reindex(node_order, fill_value=0.0).values
    mask       = means >= min_mean_pass
    n_active   = int(mask.sum())
    n_excluido = len(node_order) - n_active
    log.info(
        f"active_mask: criterio demanda_media >= {min_mean_pass} pass./janela | "
        f"nos ativos: {n_active:,} / {len(node_order):,} "
        f"({n_excluido:,} excluidos = {n_excluido/len(node_order)*100:.1f}%)"
    )
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Pesos de demanda — escala real (train_mean.csv)
# ─────────────────────────────────────────────────────────────────────────────

def build_demand_weights(train_mean_path: str, node_order: list, cfg) -> np.ndarray:
    """
    Pesos de demanda w_n = (mean_n + eps)^beta, normalizados para media=1.
    Calculados sobre train_mean.csv (escala original de passageiros).
    Retorna np.ndarray float32 de shape (N,) para serializacao no worker.
    """
    train_mean = pd.read_csv(train_mean_path, index_col=0)
    means      = train_mean["loading"].reindex(node_order, fill_value=0.0).values.astype(np.float32)

    n_missing = int((train_mean["loading"].reindex(node_order).isna()).sum())
    if n_missing > 0:
        log.warning(
            f"build_demand_weights: {n_missing} nos ausentes em train_mean.csv "
            f"— media imputada como 0.0 (peso minimo)."
        )

    weights = (means + 1e-6) ** cfg.demand_weight_beta
    weights = (weights / weights.mean()).astype(np.float32)
    log.info(
        f"Pesos demanda: beta={cfg.demand_weight_beta} | "
        f"min={weights.min():.3f} | max={weights.max():.3f} | "
        f"razao={weights.max()/weights.min():.1f}x | fonte: train_mean.csv"
    )
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Resolucao de node_order com fallback em 3 niveis
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_node_order(checkpoint_data, train_mean: pd.DataFrame,
                         artifacts_dir: str, N: int):
    """
    Nivel 1: checkpoint contem 'node_order' — valida e usa diretamente.
    Nivel 2: sem node_order, node_metadata.csv disponivel — reconstroi.
    Nivel 3: fallback para stop_order de train_mean.csv com WARNING.

    Retorna (node_order: list, fonte: str).
    """
    stop_order = list(train_mean.index)

    ckpt_node_order = (
        checkpoint_data.get("node_order")
        if isinstance(checkpoint_data, dict) else None
    )
    if ckpt_node_order is not None:
        if len(ckpt_node_order) != N:
            raise ValueError(
                f"node_order do checkpoint tem {len(ckpt_node_order)} nos, "
                f"mas tensor tem N={N}."
            )
        set_ckpt  = set(ckpt_node_order)
        set_mean  = set(stop_order)
        only_ckpt = set_ckpt - set_mean
        only_mean = set_mean - set_ckpt
        if only_ckpt or only_mean:
            raise ValueError(
                f"Incompatibilidade de node_ids entre checkpoint e train_mean.csv:\n"
                f"  Apenas no checkpoint: {len(only_ckpt)} nos\n"
                f"  Apenas no train_mean: {len(only_mean)} nos"
            )
        n_diff = sum(a != b for a, b in zip(ckpt_node_order, stop_order))
        if n_diff > 0:
            log.warning(
                f"node_order e stop_order tem mesmo conjunto mas {n_diff:,} "
                f"posicoes diferem. desnormalize() usa .reindex() — OK."
            )
        else:
            log.info("Alinhamento node_order OK.")
        return ckpt_node_order, "checkpoint[node_order]"

    meta_path = os.path.join(artifacts_dir, "node_metadata.csv")
    if os.path.exists(meta_path):
        log.warning("Checkpoint sem 'node_order'. Reconstruindo de node_metadata.csv...")
        try:
            meta = pd.read_csv(meta_path)
            if "node_id" in meta.columns and "node_index" in meta.columns:
                meta_sorted = meta.sort_values("node_index").reset_index(drop=True)
                node_order  = meta_sorted["node_id"].tolist()
                if len(node_order) == N:
                    set_meta = set(node_order)
                    set_mean = set(stop_order)
                    if not (set_meta - set_mean) and not (set_mean - set_meta):
                        return node_order, "node_metadata.csv"
        except Exception as exc:
            log.warning(f"Erro ao ler node_metadata.csv: {exc}.")

    log.warning(
        "FALLBACK: usando stop_order de train_mean.csv como node_order. "
        "Alinhamento nao garantido. Retreine com train.py atualizado."
    )
    return stop_order, "train_mean.csv (fallback)"


# ─────────────────────────────────────────────────────────────────────────────
# Metricas e auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    sparse_threshold: int = 2) -> dict:
    with np.errstate(all="ignore"):
        mae   = float(np.nanmean(np.abs(pred - target)))
        rmse  = float(np.sqrt(np.nanmean((pred - target) ** 2)))
        mask  = target > sparse_threshold
        mape  = (
            float(np.nanmean(np.abs((pred[mask] - target[mask]) / target[mask])) * 100)
            if mask.any() else float("nan")
        )
        smape = float(
            np.nanmean(
                2 * np.abs(pred - target) / (np.abs(pred) + np.abs(target) + 1e-8)
            ) * 100
        )
    return {"MAE": mae, "RMSE": rmse, "MAPE_filt": mape, "sMAPE": smape}


def desnormalize(values: np.ndarray, mean_series: pd.Series,
                 std_series: pd.Series, node_order: list) -> np.ndarray:
    missing = [sid for sid in node_order if sid not in mean_series.index]
    if missing:
        log.warning(
            f"{len(missing)} paradas ausentes em train_mean/std: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    means = mean_series.reindex(node_order, fill_value=0.0).values
    stds  = std_series.reindex(node_order,  fill_value=1.0).values
    return values * stds[None, None, :] + means[None, None, :]


def stratify_stops(targets: np.ndarray) -> dict:
    with np.errstate(all="ignore"):
        mean_per_stop = np.nanmean(targets, axis=(0, 1))
    return {
        "Baixa (<5)":   np.where(mean_per_stop < 5)[0],
        "Media (5-20)": np.where((mean_per_stop >= 5) & (mean_per_stop < 20))[0],
        "Alta (>=20)":  np.where(mean_per_stop >= 20)[0],
    }


def free_gpu(*models) -> None:
    for m in models:
        if m is not None and isinstance(m, torch.nn.Module):
            try:
                m.cpu()
            except Exception:
                pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Worker do processo filho — AGCRN em contexto CUDA limpo
# ─────────────────────────────────────────────────────────────────────────────

def _agcrn_worker(artifacts_dir: str, checkpoint_dir: str,
                  cfg_dict: dict, demand_weights_np: np.ndarray,
                  node_order: list, output_path: str, log_queue) -> None:
    """
    Executa AGCRN em processo filho com contexto CUDA isolado.
    node_order passado explicitamente para persistencia no checkpoint.
    Splits carregados separadamente para minimizar pico de RAM.
    """
    import logging
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s | %(levelname)s | [AGCRN worker] %(message)s",
        handlers = [logging.StreamHandler()],
    )
    child_log = logging.getLogger("agcrn_worker")

    if cfg_dict["src"] not in sys.path:
        sys.path.insert(0, cfg_dict["src"])

    # Adiciona baselines/ ao path do filho para localizar agcrn.py.
    # O processo filho (spawn) nao herda sys.path do pai no Windows.
    baselines_dir = cfg_dict.get("baselines_dir", "")
    if baselines_dir and baselines_dir not in sys.path:
        sys.path.insert(0, baselines_dir)

    from config  import Config
    from dataset import make_loaders
    from agcrn   import AGCRNWrapper

    try:
        cfg           = Config()
        cfg.num_nodes = cfg_dict["num_nodes"]
        # [A9] Sobrescreve lr, epochs e patience com valores do processo pai.
        cfg.lr        = cfg_dict["lr"]
        cfg.epochs    = cfg_dict["epochs"]
        cfg.patience  = cfg_dict["patience"]

        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = cfg.use_amp and device.type == "cuda"
        child_log.info(f"Dispositivo: {device} | AMP: {use_amp}")

        if torch.cuda.is_available():
            free_mb = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated(0)
            ) / 1024 ** 2
            child_log.info(f"VRAM livre (contexto limpo): {free_mb:.0f} MB")

        X_train = np.load(os.path.join(artifacts_dir, "X_train.npy"))
        X_val   = np.load(os.path.join(artifacts_dir, "X_val.npy"))
        train_loader, val_loader, _ = make_loaders(X_train, X_val, None, cfg)
        del X_train, X_val
        gc.collect()

        X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
        _, _, test_loader = make_loaders(None, None, X_test, cfg)
        del X_test
        gc.collect()

        demand_weights = torch.from_numpy(demand_weights_np)

        wrapper = AGCRNWrapper(
            cfg=cfg, device=device, use_amp=use_amp,
            demand_weights=demand_weights,
            node_order=node_order,
        )
        wrapper.train(train_loader, val_loader, checkpoint_dir)
        preds, targets = wrapper.infer(test_loader)

        np.savez(output_path, preds=preds, targets=targets)
        child_log.info(f"Resultados salvos em: {output_path}.npz")

    except Exception as e:
        child_log.error(f"Erro no processo filho: {e}", exc_info=True)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Inferencia — GraphWaveNet
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_gwn(model: torch.nn.Module, test_loader,
              device: torch.device, use_amp: bool) -> tuple:
    model.eval()
    all_preds, all_targets = [], []
    device_type = device.type

    for X, y in test_loader:
        X = X.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            pred = model(X).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y.numpy())
        del X, y

    return (
        np.concatenate(all_preds,   axis=0).astype(np.float32),
        np.concatenate(all_targets, axis=0).astype(np.float32),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Relatorio comparativo
# ─────────────────────────────────────────────────────────────────────────────

def _cache_metrics(results: dict, cfg) -> dict:
    strata_ref = stratify_stops(list(results.values())[0]["targets"])
    cache = {}
    for name, res in results.items():
        cache[name] = {"global": compute_metrics(res["preds"], res["targets"])}
        for h in range(cfg.out_steps):
            cache[name][h] = compute_metrics(
                res["preds"][:, h, :], res["targets"][:, h, :]
            )
        cache[name]["strata"] = {}
        for group, idx in strata_ref.items():
            cache[name]["strata"][group] = (
                None if len(idx) == 0
                else compute_metrics(
                    res["preds"][:, :, idx], res["targets"][:, :, idx]
                )
            )
    return cache


def print_comparison(results: dict, cfg,
                     n_active: int, n_total: int) -> None:
    """Relatorio comparativo. horizons via cfg.horizons_min — sem hardcode."""
    names    = list(results.keys())
    horizons = cfg.horizons_min
    cache    = _cache_metrics(results, cfg)

    log.info(f"\n{_SEP_WIDE}")
    log.info("  COMPARACAO -- METRICAS GLOBAIS POR HORIZONTE")
    log.info(f"  nos ativos: {n_active:,} / {n_total:,} | freq_min={cfg.freq_min}min")
    log.info(_SEP_WIDE)

    for metric in ("MAE", "RMSE", "sMAPE"):
        log.info(f"\n  {metric}")
        log.info(f"  {'Horizonte':>10}" + "".join(f"{n:>16}" for n in names))
        log.info(f"  {'-'*10}" + "-" * 16 * len(names))
        for h, hz in enumerate(horizons):
            row = f"  {hz:>7} min" + "".join(
                f"  {cache[name][h][metric]:>12.3f}  " for name in names
            )
            log.info(row)
        log.info(
            f"  {'Global':>10}" + "".join(
                f"  {cache[name]['global'][metric]:>12.3f}  " for name in names
            )
        )

    strata = stratify_stops(list(results.values())[0]["targets"])
    log.info(f"\n{_SEP_WIDE}")
    log.info("  COMPARACAO POR ESTRATO -- MAE Global  [nos ativos]")
    log.info(_SEP_WIDE)
    log.info(f"  {'Estrato':>18}" + "".join(f"{n:>16}" for n in names))
    log.info(f"  {'-'*18}" + "-" * 16 * len(names))
    for group, idx in strata.items():
        if len(idx) == 0:
            continue
        row = f"  {group:>18}"
        for name in names:
            m = cache[name]["strata"].get(group)
            row += f"  {m['MAE']:>12.3f}  " if m else f"  {'n/a':>12}  "
        log.info(row)

    if "GraphWaveNet" in results and len(names) > 1:
        log.info(f"\n{_SEP_WIDE}")
        log.info("  REDUCAO DE MAE GLOBAL: GraphWaveNet vs. AGCRN")
        log.info(_SEP_WIDE)
        gwn_mae = cache["GraphWaveNet"]["global"]["MAE"]
        for name in names:
            if name == "GraphWaveNet":
                continue
            base_mae = cache[name]["global"]["MAE"]
            gain     = (base_mae - gwn_mae) / base_mae * 100
            sign     = "GWN melhor" if gain > 0 else "AGCRN melhor"
            log.info(
                f"  vs. {name:<22}: {gain:+.2f}%  "
                f"(AGCRN={base_mae:.4f}, GWN={gwn_mae:.4f})  [{sign}]"
            )


def save_results_csv(results: dict, cfg, artifacts_dir: str) -> None:
    """horizons via cfg.horizons_min — sem hardcode."""
    rows     = []
    horizons = cfg.horizons_min
    for name, res in results.items():
        m = compute_metrics(res["preds"], res["targets"])
        rows.append({"model": name, "horizonte_min": "Global", **m})
        for h, hz in enumerate(horizons):
            m = compute_metrics(res["preds"][:, h, :], res["targets"][:, h, :])
            rows.append({"model": name, "horizonte_min": hz, **m})
    path = os.path.join(artifacts_dir, "baseline_comparison.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"Resultados salvos em: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_baselines():
    from config  import Config
    from dataset import make_loaders
    from model   import GraphWaveNet

    cfg     = Config()
    device  = torch.device(
        "cuda" if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"
    )
    use_amp = cfg.use_amp and device.type == "cuda"

    log.info(f"Dispositivo : {device} | AMP: {use_amp}")
    log.info(f"sources/    : {SRC}")
    log.info(f"Janela      : {cfg.freq_min} min | horizons: {cfg.horizons_min}")
    log.info(
        f"Perda       : weighted_masked_MAE_multihorizon "
        f"(beta={cfg.demand_weight_beta}, hw={cfg.horizon_weights})"
    )
    log.info(
        f"Treinamento : batch={cfg.batch_size} | epochs={cfg.epochs} | "
        f"patience={cfg.patience} | lr={cfg.lr}"
    )

    X_test   = np.load(os.path.join(cfg.ARTIFACTS_DIR, "X_test.npy"))
    adj_geo  = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_geo.npy"))
    adj_topo = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_topo.npy"))

    cfg.num_nodes = X_test.shape[1]
    N = cfg.num_nodes
    log.info(f"Paradas (N) : {N:,} | Features (F): {X_test.shape[2]}")

    train_mean = pd.read_csv(os.path.join(cfg.ARTIFACTS_DIR, "train_mean.csv"), index_col=0)
    train_std  = pd.read_csv(os.path.join(cfg.ARTIFACTS_DIR, "train_std.csv"),  index_col=0)

    # node_order canonico de node_metadata.csv (fonte mais confiavel antes dos checkpoints)
    meta_path = os.path.join(cfg.ARTIFACTS_DIR, "node_metadata.csv")
    if os.path.exists(meta_path):
        node_meta      = pd.read_csv(meta_path).sort_values("node_index")
        node_order_pre = node_meta["node_id"].tolist()
        assert len(node_order_pre) == N, (
            f"node_metadata.csv tem {len(node_order_pre)} nos, tensor tem N={N}."
        )
    else:
        log.warning(
            "node_metadata.csv ausente — usando stop_order de train_mean.csv "
            "para node_order inicial. Pode causar desalinhamento."
        )
        node_order_pre = list(train_mean.index)

    # active_mask em escala real — criterio interpretavel em passageiros
    active_mask = build_active_mask(train_mean, node_order_pre, min_mean_pass=1.0)
    n_active    = int(active_mask.sum())
    log.info(f"Nos ativos  : {n_active:,} / {N:,}")

    train_mean_path   = os.path.join(cfg.ARTIFACTS_DIR, "train_mean.csv")
    demand_weights_np = build_demand_weights(train_mean_path, node_order_pre, cfg)

    results = {}

    # ── 1. AGCRN em processo filho isolado ────────────────────────────────────
    log.info(f"\n{_SEP_THIN}")
    log.info("[1/2] AGCRN (Bai et al., NeurIPS 2020)")
    log.info("  Executando em processo filho isolado (contexto CUDA limpo).")
    log.info(_SEP_THIN)

    # mkstemp SEM suffix: np.savez() acrescenta ".npz" automaticamente,
    # evitando extensao dupla "output.npz.npz" que impedia np.load() de
    # encontrar o arquivo.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix="", prefix="agcrn_results_")
    os.close(tmp_fd)
    os.unlink(tmp_path)

    # baselines_dir: diretorio que contem agcrn.py.
    # O worker roda em processo filho com sys.path limpo — precisa receber
    # o caminho explicitamente para localizar agcrn.py, independente de onde
    # compare_plot_results.py esta sendo executado (sources/ ou baselines/).
    _script_dir   = os.path.dirname(os.path.abspath(__file__))
    _baselines_dir = os.path.normpath(os.path.join(_script_dir, "..", "baselines"))
    if not os.path.isfile(os.path.join(_baselines_dir, "agcrn.py")):
        # Fallback: agcrn.py no mesmo diretorio do script
        _baselines_dir = _script_dir
    # [A9] lr, epochs e patience passados explicitamente no cfg_dict.
    # lr=1e-4 para AGCRN: lr=1e-3 leva o modelo ao platô de prever
    # a media na época 1 com N=13.642 — LR menor permite explorar
    # a curvatura da loss antes de fixar nos embeddings DAGG.
    cfg_dict = {
        "src":          SRC,
        "baselines_dir": _baselines_dir,
        "num_nodes":    N,
        "epochs":       cfg.epochs,    # [A9] 150
        "patience":     cfg.patience,  # [A9] 20
        "lr":           1e-4,          # [A9] reduzido de 1e-3 para AGCRN
    }

    ctx     = mp.get_context("spawn")
    process = ctx.Process(
        target = _agcrn_worker,
        args   = (cfg.ARTIFACTS_DIR, cfg.CHECKPOINT_DIR,
                  cfg_dict, demand_weights_np,
                  node_order_pre, tmp_path, None),
        daemon = False,
    )
    process.start()
    log.info(f"  Processo filho iniciado (PID={process.pid}).")
    process.join()

    if process.exitcode != 0:
        raise RuntimeError(
            f"Processo filho do AGCRN encerrou com codigo {process.exitcode}."
        )

    log.info(f"  Processo filho encerrado (PID={process.pid}). VRAM liberada pelo SO.")

    npz_path = tmp_path + ".npz"
    # Gerenciador de contexto garante fechamento do mmap antes do unlink
    # (necessario no Windows para evitar PermissionError ao deletar arquivo mapeado).
    with np.load(npz_path) as data:
        p_ag = data["preds"].copy()
        t_ag = data["targets"].copy()
    os.unlink(npz_path)

    agcrn_ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "agcrn_best.pt")
    if os.path.exists(agcrn_ckpt_path):
        agcrn_ckpt = torch.load(agcrn_ckpt_path, map_location="cpu", weights_only=False)
        node_order_ag, src_ag = _resolve_node_order(
            agcrn_ckpt, train_mean, cfg.ARTIFACTS_DIR, N
        )
        log.info(f"  AGCRN node_order: {src_ag}")
    else:
        node_order_ag = node_order_pre
        log.warning("  AGCRN checkpoint nao encontrado — usando node_order de node_metadata.csv.")

    def denorm_ag(arr: np.ndarray) -> np.ndarray:
        return desnormalize(arr, train_mean["loading"], train_std["loading"], node_order_ag)

    results["AGCRN"] = {
        "preds":   denorm_ag(p_ag)[:, :, active_mask],
        "targets": denorm_ag(t_ag)[:, :, active_mask],
    }
    log.info(f"  MAE normalizado: {np.nanmean(np.abs(p_ag - t_ag)):.4f}")
    del p_ag, t_ag
    gc.collect()

    # ── 2. GraphWaveNet ───────────────────────────────────────────────────────
    log.info(f"\n{_SEP_THIN}")
    log.info("[2/2] GraphWaveNet (checkpoint salvo)")
    log.info(_SEP_THIN)

    # Apenas test_loader necessario — X_train e X_val nao carregados
    _, _, test_loader = make_loaders(None, None, X_test, cfg)

    gwn       = GraphWaveNet(cfg, adj_geo, adj_topo).to(device)
    ckpt_data = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        # Valida in_features antes de load_state_dict — erro explicito
        ckpt_in_features = ckpt_data.get("in_features")
        if ckpt_in_features is not None and int(ckpt_in_features) != cfg.in_features:
            raise ValueError(
                f"Checkpoint GWN treinado com in_features={ckpt_in_features}, "
                f"mas cfg.in_features={cfg.in_features} e tensor tem F={X_test.shape[2]}."
            )

        gwn.load_state_dict(ckpt_data["model_state"])
        log.info(
            f"  Checkpoint: epoca {ckpt_data.get('epoch','?')} | "
            f"val MAE: {ckpt_data.get('best_val', float('nan')):.4f} | "
            f"freq_min={ckpt_data.get('freq_min','?')}min"
        )
        ckpt_freq = ckpt_data.get("freq_min", "?")
        if ckpt_freq != "?" and int(ckpt_freq) != cfg.freq_min:
            log.warning(
                f"  ATENCAO: checkpoint treinado com freq_min={ckpt_freq}min, "
                f"mas cfg.freq_min={cfg.freq_min}min."
            )
    else:
        gwn.load_state_dict(ckpt_data)
        log.info("  Checkpoint legado (state_dict direto).")

    node_order_gw, src_gw = _resolve_node_order(
        ckpt_data, train_mean, cfg.ARTIFACTS_DIR, N
    )
    log.info(f"  GWN node_order: {src_gw} ({len(node_order_gw):,} nos)")

    def denorm_gw(arr: np.ndarray) -> np.ndarray:
        return desnormalize(arr, train_mean["loading"], train_std["loading"], node_order_gw)

    p_gw, t_gw = infer_gwn(gwn, test_loader, device, use_amp)
    results["GraphWaveNet"] = {
        "preds":   denorm_gw(p_gw)[:, :, active_mask],
        "targets": denorm_gw(t_gw)[:, :, active_mask],
    }
    log.info(f"  MAE normalizado: {np.nanmean(np.abs(p_gw - t_gw)):.4f}")
    free_gpu(gwn)
    del gwn, p_gw, t_gw

    print_comparison(results, cfg, n_active, N)
    save_results_csv(results, cfg, cfg.ARTIFACTS_DIR)

    return results


if __name__ == "__main__":
    mp.freeze_support()
    run_baselines()