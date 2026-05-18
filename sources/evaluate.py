import os
import logging
import sys

import numpy as np
import pandas as pd
import torch

from config  import Config
from model   import GraphWaveNet
from dataset import make_loaders

_stream_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8",
                closefd=False, buffering=1)
)

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler("evaluation.log", encoding="utf-8"),
        _stream_handler,
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Mascara de nos ativos
# ─────────────────────────────────────────────────────────────────────────────

def build_active_mask(train_mean: pd.DataFrame,
                      train_std:  pd.DataFrame,
                      node_order: list,
                      min_mean_pass: float = 1.0) -> np.ndarray:
    """
    Identifica nos com dado real suficiente usando escala original (passageiros).

    Criterio: demanda media no treino >= min_mean_pass passageiros/janela.
    Criterio interpretavel em unidades reais, em vez de percentil sobre
    variancia do tensor z-score (que nao tem significado operacional).

    min_mean_pass=1.0: exclui paradas com demanda media < 1 pass./janela,
    correspondendo a nos majoritariamente imputados ou sem operacao real.
    Consistente com min_obs_frac=0.25 aplicado no pipeline.
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
                f"  Apenas no train_mean: {len(only_mean)} nos\n"
                f"O checkpoint e train_mean.csv pertencem a execucoes incompativeis."
            )
        n_diff = sum(a != b for a, b in zip(ckpt_node_order, stop_order))
        if n_diff > 0:
            log.warning(
                f"node_order e stop_order tem mesmo conjunto mas {n_diff:,} "
                f"posicoes diferem em ordem. desnormalize() usa .reindex() — OK."
            )
        else:
            log.info("Alinhamento node_order OK — checkpoint e train_mean.csv consistentes.")
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
# Metricas e estratificacao
# ─────────────────────────────────────────────────────────────────────────────

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


def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    sparse_threshold: int = 2) -> dict:
    with np.errstate(all="ignore"):
        mae  = float(np.nanmean(np.abs(pred - target)))
        rmse = float(np.sqrt(np.nanmean((pred - target) ** 2)))
        mask_filt = target > sparse_threshold
        mape_filt = (
            float(np.nanmean(
                np.abs((pred[mask_filt] - target[mask_filt]) / target[mask_filt])
            ) * 100)
            if mask_filt.any() else float("nan")
        )
        smape = float(
            np.nanmean(
                2 * np.abs(pred - target) / (np.abs(pred) + np.abs(target) + 1e-8)
            ) * 100
        )
        mask_nz = target > 0
        mae_nz  = (
            float(np.nanmean(np.abs(pred[mask_nz] - target[mask_nz])))
            if mask_nz.any() else float("nan")
        )
    return {
        "MAE":         mae,
        "RMSE":        rmse,
        "MAPE_filt":   mape_filt,
        "sMAPE":       smape,
        "MAE_nonzero": mae_nz,
    }


def stratify_stops(targets: np.ndarray) -> dict:
    with np.errstate(all="ignore"):
        mean_per_stop = np.nanmean(targets, axis=(0, 1))
    return {
        "Baixa (<5)":   np.where(mean_per_stop < 5)[0],
        "Media (5-20)": np.where((mean_per_stop >= 5) & (mean_per_stop < 20))[0],
        "Alta (>=20)":  np.where(mean_per_stop >= 20)[0],
    }


def stratify_alta(targets: np.ndarray) -> dict:
    """
    Sub-estratificacao do estrato Alta (>=20 pass.) em tres faixas.
    Limiares baseados na distribuicao observada (demanda max ~104 pass.):
      Alta-baixa : 20-40 pass.
      Alta-media : 40-70 pass.
      Alta-alta  : >70 pass. (cauda de alta variancia)
    """
    with np.errstate(all="ignore"):
        mean_per_stop = np.nanmean(targets, axis=(0, 1))
    return {
        "Alta-baixa (20-40)": np.where((mean_per_stop >= 20) & (mean_per_stop < 40))[0],
        "Alta-media (40-70)": np.where((mean_per_stop >= 40) & (mean_per_stop < 70))[0],
        "Alta-alta  (>=70)":  np.where(mean_per_stop >= 70)[0],
    }


def diagnostico_inversao(preds: np.ndarray, targets: np.ndarray,
                          horizons: list, label: str = "") -> None:
    """
    Delta MAE entre horizontes consecutivos.
    delta > 0: MAE cresce com o horizonte (esperado).
    delta < 0: MAE decresce — padrao invertido (anomalia).
    """
    with np.errstate(all="ignore"):
        maes = [float(np.nanmean(np.abs(preds[:, h, :] - targets[:, h, :])))
                for h in range(len(horizons))]
    partes = []
    for i in range(1, len(horizons)):
        delta = maes[i] - maes[i - 1]
        sinal = "crescente (OK)" if delta > 0 else "DECRESCENTE (invertido)"
        partes.append(
            f"t+{horizons[i-1]}->t+{horizons[i]}min: delta={delta:+.3f} [{sinal}]"
        )
    prefixo = f"  [{label}] " if label else "  "
    log.info(prefixo + " | ".join(partes))


# ─────────────────────────────────────────────────────────────────────────────
# Formatacao de log
# ─────────────────────────────────────────────────────────────────────────────

_SEP = "-" * 66


def _section(title: str) -> None:
    log.info(f"\n{_SEP}\n  {title}\n{_SEP}")


def _header() -> None:
    log.info(
        f"{'Horizonte':>12} {'MAE':>8} {'RMSE':>8} "
        f"{'MAPE_filt':>11} {'sMAPE':>8} {'MAE_nz':>8}"
    )
    log.info(f"{'-'*12} {'-'*8} {'-'*8} {'-'*11} {'-'*8} {'-'*8}")


def _row(label: str, m: dict) -> None:
    mape   = f"{m['MAPE_filt']:>9.2f}%" if not np.isnan(m["MAPE_filt"]) else f"{'n/a':>10}"
    mae_nz = f"{m['MAE_nonzero']:>8.3f}" if not np.isnan(m["MAE_nonzero"]) else f"{'n/a':>8}"
    log.info(
        f"{label:>12}   {m['MAE']:>7.3f}   {m['RMSE']:>7.3f} "
        f"{mape}   {m['sMAPE']:>7.2f}%   {mae_nz}"
    )


def _global_row(m: dict) -> None:
    log.info(
        f"\n{'Global':>12}   {m['MAE']:>7.3f}   {m['RMSE']:>7.3f}"
        f"   {m['MAPE_filt']:>8.2f}%   {m['sMAPE']:>7.2f}%"
        f"   {m['MAE_nonzero']:>7.3f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Avaliacao principal
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation():
    cfg         = Config()
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    use_amp     = cfg.use_amp and device_type == "cuda"

    log.info(f"Dispositivo: {device} | AMP: {use_amp}")
    log.info(
        f"Janela: {cfg.freq_min} min | "
        f"in_steps={cfg.in_steps} | out_steps={cfg.out_steps} | "
        f"horizons={cfg.horizons_min}"
    )

    X_test   = np.load(os.path.join(cfg.ARTIFACTS_DIR, "X_test.npy"))
    adj_geo  = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_geo.npy"))
    adj_topo = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_topo.npy"))

    cfg.num_nodes = X_test.shape[1]
    N = cfg.num_nodes
    log.info(f"Paradas (N): {N} | Features (F): {X_test.shape[2]}")

    assert X_test.shape[2] == cfg.in_features, (
        f"in_features no tensor de teste ({X_test.shape[2]}) != cfg ({cfg.in_features}). "
        f"Verifique ALL_FEATURES no pipeline e in_features em config.py."
    )

    _, _, test_loader = make_loaders(None, None, X_test, cfg)

    model = GraphWaveNet(cfg, adj_geo, adj_topo).to(device)

    checkpoint_data = torch.load(
        cfg.checkpoint_path, map_location=device, weights_only=False
    )

    if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
        ckpt_in_features = checkpoint_data.get("in_features")
        if ckpt_in_features is not None and int(ckpt_in_features) != cfg.in_features:
            raise ValueError(
                f"Checkpoint treinado com in_features={ckpt_in_features}, "
                f"mas cfg.in_features={cfg.in_features} e tensor tem F={X_test.shape[2]}."
            )
        model.load_state_dict(checkpoint_data["model_state"])
        saved_epoch = checkpoint_data.get("epoch", "?")
        best_val    = checkpoint_data.get("best_val", float("nan"))
        ckpt_freq   = checkpoint_data.get("freq_min", "?")
        log.info(
            f"Checkpoint: epoca {saved_epoch} | best val MAE: {best_val:.4f} | "
            f"freq_min={ckpt_freq}min"
        )
        if ckpt_freq != "?" and int(ckpt_freq) != cfg.freq_min:
            log.warning(
                f"ATENCAO: checkpoint treinado com freq_min={ckpt_freq}min, "
                f"mas cfg.freq_min={cfg.freq_min}min."
            )
    else:
        model.load_state_dict(checkpoint_data)
        log.info("Checkpoint legado carregado (apenas state_dict).")

    model.eval()

    train_mean = pd.read_csv(
        os.path.join(cfg.ARTIFACTS_DIR, "train_mean.csv"), index_col=0
    )
    train_std  = pd.read_csv(
        os.path.join(cfg.ARTIFACTS_DIR, "train_std.csv"), index_col=0
    )

    node_order, node_order_src = _resolve_node_order(
        checkpoint_data, train_mean, cfg.ARTIFACTS_DIR, N
    )
    log.info(f"node_order fonte: {node_order_src} ({len(node_order):,} nos)")

    active_mask = build_active_mask(train_mean, train_std, node_order, min_mean_pass=1.0)
    n_active    = int(active_mask.sum())
    n_excluido  = N - n_active

    all_preds, all_targets = [], []
    for X, y in test_loader:
        X = X.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device_type, enabled=use_amp):
            pred = model(X).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y.numpy())
        del X, y

    preds   = np.concatenate(all_preds,   axis=0)
    targets = np.concatenate(all_targets, axis=0)

    preds_real   = desnormalize(preds,   train_mean["loading"], train_std["loading"], node_order)
    targets_real = desnormalize(targets, train_mean["loading"], train_std["loading"], node_order)

    n_nan = int(np.isnan(preds_real).any(axis=(0, 1)).sum())
    if n_nan > 0:
        log.warning(f"{n_nan} paradas com NaN apos desnormalizacao — ignoradas via nanmean.")

    preds_a   = preds_real[:, :, active_mask]
    targets_a = targets_real[:, :, active_mask]

    horizons = cfg.horizons_min

    _section(f"METRICAS GLOBAIS POR HORIZONTE  [nos ativos: {n_active:,} / {N:,}]")
    _header()
    for h, hz in enumerate(horizons):
        _row(f"{hz} min", compute_metrics(preds_a[:, h, :], targets_a[:, h, :]))
    _global_row(compute_metrics(preds_a, targets_a))

    strata = stratify_stops(targets_a)
    for group, idx in strata.items():
        n = len(idx)
        _section(f"ESTRATO -- {group}  ({n} paradas / {n/n_active*100:.1f}% dos ativos)")
        if n == 0:
            log.info("  Nenhuma parada neste estrato.")
            continue
        _header()
        for h, hz in enumerate(horizons):
            _row(f"{hz} min", compute_metrics(preds_a[:, h, idx], targets_a[:, h, idx]))
        _global_row(compute_metrics(preds_a[:, :, idx], targets_a[:, :, idx]))

    _section("DIAGNOSTICO — PADRAO DE MAE ENTRE HORIZONTES")
    log.info("  delta > 0: MAE cresce com horizonte (esperado)")
    log.info("  delta < 0: MAE decresce com horizonte (anomalia)")
    log.info("")

    diagnostico_inversao(preds_a, targets_a, horizons, "Global")
    for group, idx in strata.items():
        if len(idx) == 0:
            continue
        diagnostico_inversao(preds_a[:, :, idx], targets_a[:, :, idx], horizons, group)

    log.info("")
    log.info("  Sub-estratos do estrato Alta (>=20 pass.):")
    sub = stratify_alta(targets_a)
    for group, idx in sub.items():
        n = len(idx)
        if n == 0:
            log.info(f"  [{group}] 0 paradas — estrato vazio.")
            continue
        with np.errstate(all="ignore"):
            d               = np.nanmean(targets_a[:, :, idx], axis=(0, 1))
            variancia_media = float(np.nanmean(np.nanvar(targets_a[:, :, idx], axis=0)))
        diagnostico_inversao(
            preds_a[:, :, idx], targets_a[:, :, idx], horizons,
            f"{group} | n={n} | demanda_media={np.nanmean(d):.1f} | var_media={variancia_media:.2f}"
        )

    log.info("")
    log.info("  Correlacao entre demanda media e delta MAE (h1-h0) por parada:")
    with np.errstate(all="ignore"):
        mean_per_stop  = np.nanmean(targets_a, axis=(0, 1))
        delta_mae_stop = (
            np.nanmean(np.abs(preds_a[:, 1, :] - targets_a[:, 1, :]), axis=0) -
            np.nanmean(np.abs(preds_a[:, 0, :] - targets_a[:, 0, :]), axis=0)
        )
        validos = np.isfinite(mean_per_stop) & np.isfinite(delta_mae_stop)
        if validos.sum() > 2:
            corr         = float(np.corrcoef(mean_per_stop[validos], delta_mae_stop[validos])[0, 1])
            n_invertidas = int((delta_mae_stop[validos] < 0).sum())
            n_total_v    = int(validos.sum())
            log.info(
                f"  Pearson r(demanda, delta_MAE) = {corr:+.4f} | "
                f"paradas com inversao: {n_invertidas}/{n_total_v} "
                f"({n_invertidas/n_total_v*100:.1f}%)"
            )
            if corr < -0.1:
                log.info(
                    "  Interpretacao: correlacao negativa confirma hipotese (A) — "
                    "paradas de maior demanda tem inversao mais pronunciada."
                )
            elif abs(corr) < 0.1:
                log.info(
                    "  Interpretacao: correlacao proxima de zero — inversao nao e "
                    "explicada pela demanda; investigar hipotese (B)."
                )

    _section("DISTRIBUICAO DA REDE POR ESTRATO  [nos ativos]")
    with np.errstate(all="ignore"):
        mean_per_stop = np.nanmean(targets_a, axis=(0, 1))
    for group, idx in strata.items():
        if len(idx) == 0:
            continue
        d = mean_per_stop[idx]
        log.info(
            f"  {group:16s} -- {len(idx):5d} paradas | "
            f"demanda media: {np.nanmean(d):.2f} | max: {np.nanmax(d):.2f} pass."
        )

    _section("RESUMO DA FILTRAGEM DE NOS")
    log.info(f"  Total de nos              : {N:,}")
    log.info(f"  Nos ativos (avaliados)    : {n_active:,}  ({n_active/N*100:.1f}%)")
    log.info(f"  Nos excluidos             : {n_excluido:,}  ({n_excluido/N*100:.1f}%)")
    log.info(f"  Criterio                  : demanda_media >= 1.0 pass./janela (escala real)")
    log.info(f"  Janela temporal           : {cfg.freq_min} min | horizons: {horizons}")
    log.info(f"  Todas as metricas calculadas sobre nos ativos.")

    return preds_real, targets_real, active_mask


if __name__ == "__main__":
    run_evaluation()