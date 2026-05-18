"""
run_baselines.py — versao corrigida para pipeline v7 (janela 20 min)

Alteracoes em relacao a versao anterior:

[R1] horizons via cfg.horizons_min em print_comparison e save_results_csv.
     Com freq_min=20: [20, 40] min.

[R2] Assinatura de print_comparison e save_results_csv atualizada:
     Removido parametro out_steps; adicionado cfg.

[R3] Log de horizontes e contexto temporal no inicio de run_baselines.

[R4] build_demand_weights corrigido (RB-1 / IC-3):
     A versao anterior calculava pesos sobre X_train normalizado (z-score).
     Sobre o tensor z-score, mean(loading) ~ 0 para todos os nos, tornando
     (mean + eps)^beta ~ constante — pesos praticamente uniformes, efeito
     de ponderacao por demanda nulo no AGCRN.
     train.py [T5] corrigiu isso para o GWN. run_baselines.py reintroduzia
     o bug para o AGCRN, tornando a comparacao assimetrica na funcao de perda.
     Correcao: build_demand_weights agora le train_mean.csv (escala real),
     identico a train.py [T5].

[R5] _resolve_node_order adicionado (RB-2 / IC-2):
     A versao anterior usava stop_order = list(train_mean.index) diretamente
     em desnormalize() sem validar alinhamento com o tensor.
     evaluate.py [E4] implementou _resolve_node_order() com 3 niveis de
     fallback. run_baselines.py agora usa a mesma logica, garantindo que
     desnormalize() use a ordem canonica do tensor (node_order) e nao a
     ordem arbitraria do CSV.

[R6] make_loaders corrigido no bloco GWN (RB-3):
     A chamada make_loaders(X_train, X_val, X_test, cfg) na linha do GWN
     construia train_loader e val_loader nunca iterados, mantendo X_train e
     X_val em RAM durante toda a inferencia. Com N=18.340, X_train ~ 2 GB.
     Correcao: make_loaders(None, None, X_test, cfg) — apenas test_loader.
     dataset.py [D1] suporta None e retorna None no lugar dos loaders.

[R7] Validacao de checkpoint do GWN (RB-4 / IC-P1):
     Verifica node_order e in_features do checkpoint antes de carregar
     o modelo, replicando as verificacoes [E4][E5] de evaluate.py.

[R8] horizon_weights passado ao worker do AGCRN (RB-6 / AG-1):
     O worker recebe cfg completo, que inclui cfg.horizon_weights.
     AGCRNWrapper [A5] usa hw criado de cfg.horizon_weights, portanto
     o alinhamento e automatico — nenhuma mudanca necessaria no worker,
     mas node_order e passado explicitamente para persistencia no checkpoint.

[R9] Comentario de horizons em save_results_csv corrigido (RB-5):
     Era "# ex: [15, 30]" — atualizado para "# ex: [20, 40] para freq_min=20".

[R10] Correcao da extensao dupla .npz.npz no arquivo temporario (BUG RUNTIME):
     CAUSA: tempfile.mkstemp(suffix=".npz") criava tmp_path terminado em ".npz".
     np.savez(output_path, ...) adiciona ".npz" automaticamente ao nome —
     resultado: arquivo gravado como "agcrn_results_XXXX.npz.npz".
     O processo pai tentava abrir tmp_path + ".npz" = "...npz.npz", que existia,
     mas o os.unlink(tmp_path) anterior apagava o arquivo vazio do mkstemp
     (sem extensao dupla), nao o arquivo real. Na pratica o arquivo ".npz.npz"
     era criado mas nunca encontrado por np.load porque o pai abria o caminho
     errado apos uma corrida de nomes.
     Correcao: mkstemp sem suffix (arquivo temporario sem extensao). O worker
     recebe output_path sem extensao e np.savez grava "output_path.npz"
     canonicamente. O pai le exatamente tmp_path + ".npz". Nenhuma extensao dupla.
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
# Utilitarios compartilhados
# ─────────────────────────────────────────────────────────────────────────────

def build_active_mask(X_test: np.ndarray, target_col_idx: int,
                      percentile: float = 5.0) -> np.ndarray:
    var_por_no   = X_test[:, :, target_col_idx].var(axis=0)
    var_positiva = var_por_no[var_por_no > 0]
    if len(var_positiva) == 0:
        log.warning("Todos os nos tem variancia zero -- nenhum no ativo detectado.")
        return np.zeros(X_test.shape[1], dtype=bool)
    threshold = float(np.percentile(var_positiva, percentile))
    log.info(f"active_mask: threshold (percentil {percentile}%) = {threshold:.2e}")
    return var_por_no > threshold


def build_demand_weights(train_mean_path: str, node_order: list, cfg) -> np.ndarray:
    """
    [R4] CORRECAO RB-1 / IC-3: pesos calculados sobre train_mean.csv
    (escala real de passageiros), NAO sobre X_train normalizado.

    Identico a build_demand_weights de train.py [T5].
    Retorna np.ndarray float32 de shape (N,) para serializacao no worker.
    """
    train_mean = pd.read_csv(train_mean_path, index_col=0)
    means = train_mean["loading"].reindex(node_order, fill_value=0.0).values.astype(np.float32)

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
        f"razao={weights.max()/weights.min():.1f}x | "
        f"fonte: train_mean.csv (escala original)"
    )
    return weights


def _resolve_node_order(checkpoint_data, train_mean: pd.DataFrame,
                         artifacts_dir: str, N: int):
    """
    [R5] CORRECAO RB-2 / IC-2: resolve node_order com logica em 3 niveis.
    Identico a _resolve_node_order de evaluate.py [E4].

    Nivel 1: checkpoint contem 'node_order' — valida conteudo, usa diretamente.
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
                f"  Apenas no train_mean: {len(only_mean)} nos\n"
                f"O checkpoint e train_mean.csv pertencem a execucoes incompativeis."
            )
        n_diff = sum(a != b for a, b in zip(ckpt_node_order, stop_order))
        if n_diff > 0:
            log.warning(
                f"[R5] node_order e stop_order tem mesmo conjunto mas {n_diff:,} "
                f"posicoes diferem em ordem. desnormalize() usa .reindex() — OK."
            )
        else:
            log.info("[R5] Alinhamento node_order OK.")
        return ckpt_node_order, "checkpoint[node_order]"

    meta_path = os.path.join(artifacts_dir, "node_metadata.csv")
    if os.path.exists(meta_path):
        log.warning(
            "[R5] Checkpoint sem 'node_order'. "
            "Reconstruindo de node_metadata.csv..."
        )
        try:
            meta = pd.read_csv(meta_path)
            if "node_id" in meta.columns and "node_index" in meta.columns:
                meta_sorted = meta.sort_values("node_index").reset_index(drop=True)
                node_order  = meta_sorted["node_id"].tolist()
                if len(node_order) == N:
                    set_meta  = set(node_order)
                    set_mean  = set(stop_order)
                    if not (set_meta - set_mean) and not (set_mean - set_meta):
                        log.warning(
                            "[R5] Execute repair_checkpoint.py para injetar "
                            "node_order no checkpoint e evitar esta reconstrucao."
                        )
                        return node_order, "node_metadata.csv"
        except Exception as exc:
            log.warning(f"[R5] Erro ao ler node_metadata.csv: {exc}.")

    log.warning(
        "[R5] FALLBACK: usando stop_order de train_mean.csv como node_order. "
        "Alinhamento nao garantido. Retreine com train.py atualizado."
    )
    return stop_order, "train_mean.csv (fallback)"


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
    """
    [R5] Usa node_order (ordem canonica do tensor) para .reindex().
    A ordem de train_mean.csv nao precisa coincidir com a do tensor.
    """
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
    [R8] node_order passado explicitamente para ser salvo no checkpoint
    do AGCRN [A2], simetrico com o checkpoint do GWN [T7].
    """
    import logging
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(levelname)s | [AGCRN worker] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    child_log = logging.getLogger("agcrn_worker")

    if cfg_dict["src"] not in sys.path:
        sys.path.insert(0, cfg_dict["src"])

    from config  import Config
    from dataset import make_loaders
    from agcrn   import AGCRNWrapper

    try:
        cfg           = Config()
        cfg.num_nodes = cfg_dict["num_nodes"]
        # [A9] Sobrescreve lr, epochs e patience com valores do pai.
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

        # [A7] Carrega splits separadamente para minimizar pico de RAM.
        # train + val para treino; test separado para inferencia.
        X_train = np.load(os.path.join(artifacts_dir, "X_train.npy"))
        X_val   = np.load(os.path.join(artifacts_dir, "X_val.npy"))

        # [R6] X_test=None no loader de treino/val — nao construir DataLoader desnecessario.
        train_loader, val_loader, _ = make_loaders(X_train, X_val, None, cfg)
        del X_train, X_val
        gc.collect()

        # Carrega X_test somente apos liberar train/val
        X_test = np.load(os.path.join(artifacts_dir, "X_test.npy"))
        _, _, test_loader = make_loaders(None, None, X_test, cfg)
        del X_test
        gc.collect()

        demand_weights = torch.from_numpy(demand_weights_np)

        # [R8] node_order passado para o wrapper — salvo no checkpoint [A2].
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
    """[R1] Recebe cfg. horizons = cfg.horizons_min — sem hardcode."""
    names    = list(results.keys())
    horizons = cfg.horizons_min   # [20, 40] para freq_min=20
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
    """[R1][R9] Recebe cfg. horizons = cfg.horizons_min — ex: [20, 40] para freq_min=20."""
    rows     = []
    horizons = cfg.horizons_min   # [R9] comentario corrigido: [20, 40] para freq_min=20
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

    # [R3] Log de contexto temporal
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

    active_mask = build_active_mask(X_test, cfg.target_col_idx, percentile=5.0)
    n_active    = int(active_mask.sum())
    log.info(f"Nos ativos  : {n_active:,} / {N:,} ({(N-n_active)/N*100:.1f}% excluidos)")

    train_mean = pd.read_csv(os.path.join(cfg.ARTIFACTS_DIR, "train_mean.csv"), index_col=0)
    train_std  = pd.read_csv(os.path.join(cfg.ARTIFACTS_DIR, "train_std.csv"),  index_col=0)

    # ── Carrega X_train apenas para build_demand_weights, depois libera ──────
    # [R4] Demand weights calculados de train_mean.csv, nao de X_train.
    # X_train nao e mais necessario apos isso — liberado imediatamente.
    train_mean_path   = os.path.join(cfg.ARTIFACTS_DIR, "train_mean.csv")

    # node_order sera resolvido depois do checkpoint do GWN (nivel 1)
    # ou de node_metadata.csv (nivel 2). Para build_demand_weights precisamos
    # de node_order agora — usa o node_metadata.csv como fonte canonica.
    meta_path = os.path.join(cfg.ARTIFACTS_DIR, "node_metadata.csv")
    if os.path.exists(meta_path):
        node_meta  = pd.read_csv(meta_path).sort_values("node_index")
        node_order_pre = node_meta["node_id"].tolist()
        assert len(node_order_pre) == N, (
            f"node_metadata.csv tem {len(node_order_pre)} nos, tensor tem N={N}."
        )
    else:
        # fallback: usa ordem do train_mean.csv
        log.warning(
            "node_metadata.csv ausente — usando stop_order de train_mean.csv "
            "para build_demand_weights. Pode causar pesos com alinhamento incorreto."
        )
        node_order_pre = list(train_mean.index)

    demand_weights_np = build_demand_weights(train_mean_path, node_order_pre, cfg)

    results = {}

    # ── 1. AGCRN em processo filho isolado ────────────────────────────────────
    log.info(f"\n{_SEP_THIN}")
    log.info("[1/2] AGCRN (Bai et al., NeurIPS 2020)")
    log.info("  Executando em processo filho isolado (contexto CUDA limpo).")
    log.info(_SEP_THIN)

    # [R10] mkstemp SEM suffix ".npz".
    # np.savez(output_path, ...) acrescenta ".npz" automaticamente, portanto
    # passar output_path ja terminado em ".npz" produzia extensao dupla
    # "agcrn_results_XXXX.npz.npz" — arquivo gravado com nome errado e
    # inacessivel por np.load(tmp_path + ".npz").
    # Com suffix="" o worker grava "output_path.npz" e o pai le exatamente isso.
    tmp_fd, tmp_path = tempfile.mkstemp(suffix="", prefix="agcrn_results_")
    os.close(tmp_fd)
    os.unlink(tmp_path)

    # [A9] lr, epochs e patience passados explicitamente no cfg_dict
    # para garantir que o processo filho use os valores corretos
    # independente do config.py em disco.
    # lr=1e-4 para AGCRN: lr=1e-3 leva o modelo ao platô de prever
    # a media na época 1 com N=13.642 — LR menor permite explorar
    # a curvatura da loss antes de fixar nos embeddings DAGG.
    cfg_dict = {
        "src":      SRC,
        "num_nodes": N,
        "epochs":   cfg.epochs,    # [A9] 150
        "patience": cfg.patience,  # [A9] 20
        "lr":       1e-4,          # [A9] reduzido de 1e-3 para AGCRN
    }

    ctx     = mp.get_context("spawn")
    process = ctx.Process(
        target = _agcrn_worker,
        args   = (cfg.ARTIFACTS_DIR, cfg.CHECKPOINT_DIR,
                  cfg_dict, demand_weights_np,
                  node_order_pre,     # [R8] node_order para checkpoint AGCRN
                  tmp_path, None),
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
    # np.load() no Windows abre o arquivo com memory-mapped file (mmap),
    # mantendo um file handle ativo enquanto os arrays existirem na memoria.
    # os.unlink() sobre um arquivo ainda mapeado lanca PermissionError (WinError 32)
    # no Windows — comportamento diferente do Linux, onde o unlink desvincula o
    # inode mas o handle continua valido ate ser fechado.
    # Correcao: usar gerenciador de contexto "with np.load(...) as data" garante
    # que o NpzFile.close() seja chamado antes do unlink, liberando o file handle.
    # Os arrays sao copiados para memoria (.copy()) dentro do bloco "with" para
    # desacopla-los completamente do mmap antes do fechamento.
    with np.load(npz_path) as data:
        p_ag = data["preds"].copy()
        t_ag = data["targets"].copy()
    os.unlink(npz_path)

    # Resolve node_order AGCRN via checkpoint (nivel 1) se disponivel
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

    # [R6] X_train=None e X_val=None — apenas test_loader necessario.
    _, _, test_loader = make_loaders(None, None, X_test, cfg)

    gwn       = GraphWaveNet(cfg, adj_geo, adj_topo).to(device)
    ckpt_data = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt_data, dict) and "model_state" in ckpt_data:
        # [R7] Valida in_features do checkpoint (IC-P1 / replica [E5]).
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

    # [R7] Resolve node_order do GWN (IC-P1 / replica [E4]).
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

    # [R1][R2] passa cfg em vez de out_steps
    print_comparison(results, cfg, n_active, N)
    save_results_csv(results, cfg, cfg.ARTIFACTS_DIR)

    return results


if __name__ == "__main__":
    mp.freeze_support()
    run_baselines()