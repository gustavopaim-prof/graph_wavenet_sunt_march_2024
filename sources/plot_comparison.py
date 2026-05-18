"""
plot_comparison.py
==================
Gera figuras comparativas entre AGCRN e GraphWaveNet a partir dos
resultados produzidos por compare_plot_results.py.

Pre-requisito: baseline_comparison.csv em training_data/ e os checkpoints
agcrn_best.pt e best_model.pt em checkpoints/.

Execucao:
  python sources/plot_comparison.py

Saida (results/figures/):
  C1_mae_per_horizon.pdf        — MAE por horizonte, lado a lado
  C2_rmse_per_horizon.pdf       — RMSE por horizonte, lado a lado
  C3_smape_per_horizon.pdf      — sMAPE por horizonte, lado a lado
  C4_metrics_radar.pdf          — radar chart de metricas normalizadas
  C5_mae_by_stratum.pdf         — MAE por estrato de demanda
  C6_scatter_agcrn.pdf          — scatter AGCRN predito x real
  C7_scatter_gwn.pdf            — scatter GWN predito x real
  C8_error_cdf.pdf              — CDF do erro absoluto (ambos)
  C9_error_boxplot.pdf          — boxplot do erro por horizonte (ambos)
  C10_mae_gain.pdf              — reducao relativa de MAE (AGCRN vs GWN)
  C11a/b/c_agcrn_stop1/2/3.pdf  — serie temporal AGCRN por parada
  C11a/b/c_gwn_stop1/2/3.pdf    — serie temporal GWN por parada
  C12_metrics_table.pdf         — tabela completa de metricas por horizonte e estrato
  C11a_temporal_cmp_stop1.pdf   — serie temporal comparativa (parada 1)
  C11b_temporal_cmp_stop2.pdf   — serie temporal comparativa (parada 2)
  C11c_temporal_cmp_stop3.pdf   — serie temporal comparativa (parada 3)
"""

import os
import sys
import math
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────────────────────────────────────────
# Resolucao de paths
# ─────────────────────────────────────────────────────────────────────────────

_here       = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    _here,
    os.path.normpath(os.path.join(_here, "..", "sources")),
]
SRC = None
for _c in _candidates:
    if os.path.isfile(os.path.join(_c, "config.py")):
        SRC = _c
        break
if SRC is None:
    raise RuntimeError(
        "Nao foi possivel localizar config.py.\n"
        f"  Caminhos tentados: {_candidates}"
    )
if SRC not in sys.path:
    sys.path.insert(0, SRC)

_baselines_dir = os.path.normpath(os.path.join(_here, "..", "baselines"))
if not os.path.isfile(os.path.join(_baselines_dir, "agcrn.py")):
    _baselines_dir = _here
if _baselines_dir not in sys.path:
    sys.path.insert(0, _baselines_dir)

from config  import Config
from model   import GraphWaveNet
from dataset import make_loaders
from agcrn   import AGCRNWrapper

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler("plot_comparison.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Estilo — identico ao plot_results.py (Qualis A1)
# ─────────────────────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":             9,
    "axes.titlesize":        9,
    "axes.labelsize":        9,
    "xtick.labelsize":       8,
    "ytick.labelsize":       8,
    "legend.fontsize":       8,
    "legend.title_fontsize": 8,
    "lines.linewidth":       1.2,
    "lines.markersize":      4,
    "axes.linewidth":        0.6,
    "xtick.major.width":     0.6,
    "ytick.major.width":     0.6,
    "xtick.minor.width":     0.4,
    "ytick.minor.width":     0.4,
    "xtick.direction":       "in",
    "ytick.direction":       "in",
    "xtick.minor.visible":   True,
    "ytick.minor.visible":   True,
    "axes.grid":             True,
    "grid.linestyle":        "--",
    "grid.linewidth":        0.4,
    "grid.alpha":            0.5,
    "figure.dpi":            300,
    "savefig.dpi":           300,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    "legend.framealpha":     0.9,
    "legend.edgecolor":      "0.8",
    "legend.handlelength":   1.8,
})

# Paleta daltonica (Wong 2011)
COLORS = {
    "blue":   "#0072B2",
    "orange": "#E69F00",
    "green":  "#009E73",
    "red":    "#D55E00",
    "purple": "#CC79A7",
    "sky":    "#56B4E9",
    "yellow": "#F0E442",
    "black":  "#000000",
}

MODEL_COLORS = {
    "AGCRN":        COLORS["orange"],
    "Graph Wavenet": COLORS["blue"],
}
MODEL_MARKERS = {
    "AGCRN":        "s",
    "Graph Wavenet": "o",
}

COL1 = 3.45   # largura coluna simples (in)
COL2 = 7.16   # largura coluna dupla  (in)

OUT_DIR = os.path.join(
    os.path.dirname(_here), "results", "figures"
)
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, name: str) -> None:
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    log.info(f"Salvo: {path}")
    plt.close(fig)


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
    means = mean_series.reindex(node_order, fill_value=0.0).values
    stds  = std_series.reindex(node_order,  fill_value=1.0).values
    return values * stds[None, None, :] + means[None, None, :]


def build_active_mask(train_mean: pd.DataFrame, node_order: list,
                      min_mean_pass: float = 1.0) -> np.ndarray:
    means = train_mean["loading"].reindex(node_order, fill_value=0.0).values
    mask  = means >= min_mean_pass
    log.info(
        f"active_mask: demanda_media >= {min_mean_pass} pass./janela | "
        f"nos ativos: {mask.sum():,} / {len(node_order):,}"
    )
    return mask


def stratify_stops(targets: np.ndarray) -> dict:
    with np.errstate(all="ignore"):
        mean_per_stop = np.nanmean(targets, axis=(0, 1))
    return {
        "Low\n(<5)":    np.where(mean_per_stop < 5)[0],
        "Medium\n(5–20)": np.where((mean_per_stop >= 5) & (mean_per_stop < 20))[0],
        "High\n(≥20)":  np.where(mean_per_stop >= 20)[0],
    }


def _resolve_node_order(checkpoint_data, train_mean: pd.DataFrame,
                         artifacts_dir: str, N: int) -> list:
    stop_order      = list(train_mean.index)
    ckpt_node_order = (
        checkpoint_data.get("node_order")
        if isinstance(checkpoint_data, dict) else None
    )
    if ckpt_node_order is not None and len(ckpt_node_order) == N:
        return ckpt_node_order
    meta_path = os.path.join(artifacts_dir, "node_metadata.csv")
    if os.path.exists(meta_path):
        try:
            meta       = pd.read_csv(meta_path)
            node_order = meta.sort_values("node_index")["node_id"].tolist()
            if len(node_order) == N:
                return node_order
        except Exception:
            pass
    log.warning("FALLBACK node_order: usando train_mean.csv.")
    return stop_order


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de predicoes (ambos os modelos)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def load_all_predictions() -> tuple:
    """
    Carrega predicoes de AGCRN e GWN a partir dos checkpoints salvos.
    Retorna (results, cfg, active_mask) onde:
      results = {"AGCRN": {"preds": ..., "targets": ...},
                 "Graph Wavenet": {"preds": ..., "targets": ...}}
    Todos os arrays em escala real (passageiros), filtrados por active_mask.
    """
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg.use_amp and device.type == "cuda"
    log.info(f"Dispositivo: {device} | AMP: {use_amp}")

    X_test   = np.load(os.path.join(cfg.ARTIFACTS_DIR, "X_test.npy"))
    adj_geo  = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_geo.npy"))
    adj_topo = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_topo.npy"))

    cfg.num_nodes = X_test.shape[1]
    N = cfg.num_nodes
    log.info(f"Paradas (N): {N:,} | Features (F): {X_test.shape[2]}")

    assert X_test.shape[2] == cfg.in_features, (
        f"in_features no tensor ({X_test.shape[2]}) != cfg ({cfg.in_features})."
    )

    train_mean = pd.read_csv(os.path.join(cfg.ARTIFACTS_DIR, "train_mean.csv"), index_col=0)
    train_std  = pd.read_csv(os.path.join(cfg.ARTIFACTS_DIR, "train_std.csv"),  index_col=0)

    meta_path  = os.path.join(cfg.ARTIFACTS_DIR, "node_metadata.csv")
    node_order = (
        pd.read_csv(meta_path).sort_values("node_index")["node_id"].tolist()
        if os.path.exists(meta_path) else list(train_mean.index)
    )

    active_mask = build_active_mask(train_mean, node_order, min_mean_pass=1.0)
    n_active    = int(active_mask.sum())
    log.info(f"Nos ativos: {n_active:,} / {N:,}")

    _, _, test_loader = make_loaders(None, None, X_test, cfg)

    def _infer(model):
        model.eval()
        all_p, all_t = [], []
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                all_p.append(model(X).cpu().numpy())
            all_t.append(y.numpy())
            del X, y
        return (np.concatenate(all_p, axis=0).astype(np.float32),
                np.concatenate(all_t, axis=0).astype(np.float32))

    def _to_real(p, t, no):
        pr = desnormalize(p, train_mean["loading"], train_std["loading"], no)
        tr = desnormalize(t, train_mean["loading"], train_std["loading"], no)
        return pr[:, :, active_mask], tr[:, :, active_mask]

    results = {}

    # ── AGCRN ─────────────────────────────────────────────────────────────────
    agcrn_ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, "agcrn_best.pt")
    if not os.path.exists(agcrn_ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint AGCRN nao encontrado: {agcrn_ckpt_path}\n"
            "Execute compare_plot_results.py primeiro."
        )
    agcrn_ckpt = torch.load(agcrn_ckpt_path, map_location=device, weights_only=False)
    node_order_ag = _resolve_node_order(agcrn_ckpt, train_mean, cfg.ARTIFACTS_DIR, N)

    # Reconstroi wrapper para inferencia (nao re-treina)
    demand_weights = torch.ones(N, dtype=torch.float32)
    ag_wrapper = AGCRNWrapper(
        cfg=cfg, device=device, use_amp=use_amp,
        demand_weights=demand_weights, node_order=node_order_ag,
    )
    if isinstance(agcrn_ckpt, dict) and "model_state" in agcrn_ckpt:
        ag_wrapper.model.load_state_dict(agcrn_ckpt["model_state"])
        log.info(
            f"AGCRN checkpoint: epoca {agcrn_ckpt.get('epoch','?')} | "
            f"val MAE: {agcrn_ckpt.get('best_val', float('nan')):.4f}"
        )
    ag_wrapper.model.to(device).eval()
    p_ag, t_ag = _infer(ag_wrapper.model)
    results["AGCRN"] = dict(zip(
        ("preds", "targets"), _to_real(p_ag, t_ag, node_order_ag)
    ))
    log.info(f"AGCRN MAE normalizado: {np.nanmean(np.abs(p_ag - t_ag)):.4f}")
    del p_ag, t_ag, ag_wrapper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── GraphWaveNet ──────────────────────────────────────────────────────────
    gwn_ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    node_order_gw = _resolve_node_order(gwn_ckpt, train_mean, cfg.ARTIFACTS_DIR, N)

    gwn = GraphWaveNet(cfg, adj_geo, adj_topo).to(device)
    if isinstance(gwn_ckpt, dict) and "model_state" in gwn_ckpt:
        ckpt_f = gwn_ckpt.get("in_features")
        if ckpt_f is not None and int(ckpt_f) != cfg.in_features:
            raise ValueError(
                f"GWN checkpoint in_features={ckpt_f} != cfg={cfg.in_features}."
            )
        gwn.load_state_dict(gwn_ckpt["model_state"])
        log.info(
            f"GWN checkpoint: epoca {gwn_ckpt.get('epoch','?')} | "
            f"val MAE: {gwn_ckpt.get('best_val', float('nan')):.4f}"
        )
    else:
        gwn.load_state_dict(gwn_ckpt)

    p_gw, t_gw = _infer(gwn)
    results["Graph Wavenet"] = dict(zip(
        ("preds", "targets"), _to_real(p_gw, t_gw, node_order_gw)
    ))
    log.info(f"GWN MAE normalizado: {np.nanmean(np.abs(p_gw - t_gw)):.4f}")
    gwn.cpu()
    del gwn, p_gw, t_gw
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, cfg, active_mask


# ─────────────────────────────────────────────────────────────────────────────
# C1/C2/C3 — Metricas por horizonte (linha)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_per_horizon(results: dict, cfg, metric: str,
                             ylabel: str, fname: str) -> None:
    """
    Linha por modelo, eixo x = horizontes em minutos.
    Marcadores distintos por modelo; cores da paleta padrao.
    """
    horizons = cfg.horizons_min
    fig, ax  = plt.subplots(figsize=(COL1, COL1 * 0.82))

    all_vals = []
    for name, res in results.items():
        vals = [
            compute_metrics(res["preds"][:, h, :], res["targets"][:, h, :])[metric]
            for h in range(cfg.out_steps)
        ]
        all_vals.extend(vals)
        ax.plot(horizons, vals,
                color=MODEL_COLORS[name], marker=MODEL_MARKERS[name],
                markerfacecolor="white", markeredgewidth=1.1,
                linewidth=1.3, label=name, zorder=3)
        for h, v in zip(horizons, vals):
            ax.annotate(f"{v:.2f}",
                        xy=(h, v), xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", fontsize=6.5,
                        color=MODEL_COLORS[name])

    span   = max(all_vals) - min(all_vals)
    margin = max(span * 0.45, min(all_vals) * 0.03)
    ax.set_ylim(min(all_vals) - margin, max(all_vals) + margin * 1.8)
    ax.set_xticks(horizons)
    ax.set_xlabel("Forecast horizon (min)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout(pad=0.6)
    save(fig, fname)


# ─────────────────────────────────────────────────────────────────────────────
# C4 — Radar chart de metricas normalizadas
# ─────────────────────────────────────────────────────────────────────────────

def plot_radar(results: dict, cfg) -> None:
    """
    Radar com 4 metricas globais normalizadas pelo valor do GWN (baseline).
    Quanto menor a area, melhor o modelo.
    """
    metric_labels = ["MAE", "RMSE", "MAPE_filt", "sMAPE"]
    display_labels = ["MAE", "RMSE", "MAPE\n(filtered)", "sMAPE"]

    raw = {}
    for name, res in results.items():
        m = compute_metrics(res["preds"], res["targets"])
        raw[name] = [m[k] for k in metric_labels]

    # Normaliza pelo maximo entre os dois modelos (0 = melhor, 1 = pior)
    maxvals = [max(raw[n][i] for n in raw) for i in range(len(metric_labels))]
    norm    = {n: [v / mx if mx > 0 else 0 for v, mx in zip(raw[n], maxvals)]
               for n in raw}

    n_metrics = len(metric_labels)
    angles    = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles   += angles[:1]

    fig, ax = plt.subplots(figsize=(COL1, COL1),
                            subplot_kw=dict(polar=True))

    for name, vals in norm.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=MODEL_COLORS[name], linewidth=1.2,
                marker=MODEL_MARKERS[name], markersize=5, label=name)
        ax.fill(angles, v, color=MODEL_COLORS[name], alpha=0.12)

    ax.set_thetagrids(np.degrees(angles[:-1]), display_labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6, color="gray")
    ax.spines["polar"].set_visible(False)

    # Valores absolutos anotados nos vertices para cada modelo
    for i, angle in enumerate(angles[:-1]):
        for j, (name, vals) in enumerate(norm.items()):
            r     = vals[i] + 0.10 + j * 0.07
            label = f"{raw[name][i]:.2f}"
            ax.annotate(label,
                        xy=(angle, r), fontsize=5.5,
                        color=MODEL_COLORS[name],
                        ha="center", va="center")

    ax.set_title("Normalised metric profile\n(lower = better)",
                 fontsize=8, pad=14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              ncol=2, framealpha=0.9)
    fig.tight_layout(pad=0.4)
    save(fig, "C4_metrics_radar.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# C5 — MAE por estrato de demanda
# ─────────────────────────────────────────────────────────────────────────────

def plot_mae_by_stratum(results: dict) -> None:
    """
    Barras agrupadas: eixo x = estratos de demanda, barras = modelos.
    Anotacao do valor e da reducao relativa (%) sobre cada par.
    """
    # Usa targets do primeiro modelo como referencia de estratificacao
    ref_targets = list(results.values())[0]["targets"]
    strata      = stratify_stops(ref_targets)
    names       = list(results.keys())

    labels    = list(strata.keys())
    n_strata  = len(labels)
    n_models  = len(names)
    x         = np.arange(n_strata)
    bar_w     = 0.35
    offsets   = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * bar_w

    fig, ax = plt.subplots(figsize=(COL2 * 0.65, COL2 * 0.42))

    mae_by_model = {}
    for name, res in results.items():
        mae_by_model[name] = []
        for group, idx in strata.items():
            if len(idx) == 0:
                mae_by_model[name].append(float("nan"))
            else:
                m = compute_metrics(res["preds"][:, :, idx], res["targets"][:, :, idx])
                mae_by_model[name].append(m["MAE"])

    for j, name in enumerate(names):
        bars = ax.bar(x + offsets[j], mae_by_model[name],
                      width=bar_w * 0.92, color=MODEL_COLORS[name],
                      alpha=0.82, label=name, zorder=3)
        for rect, val in zip(bars, mae_by_model[name]):
            if np.isfinite(val):
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_height() + 0.15,
                        f"{val:.2f}",
                        ha="center", va="bottom", fontsize=6.5,
                        color=MODEL_COLORS[name])

    # Reducao relativa (AGCRN vs GWN) anotada acima de cada grupo
    if "AGCRN" in mae_by_model and "Graph Wavenet" in mae_by_model:
        for i, group in enumerate(labels):
            ag_v  = mae_by_model["AGCRN"][i]
            gw_v  = mae_by_model["Graph Wavenet"][i]
            if np.isfinite(ag_v) and np.isfinite(gw_v) and gw_v > 0:
                gain = (gw_v - ag_v) / gw_v * 100
                top  = max(ag_v, gw_v) + 0.8
                ax.text(x[i], top, f"−{gain:.1f}%",
                        ha="center", va="bottom", fontsize=6.5,
                        color=COLORS["green"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Demand stratum")
    ax.set_ylabel("MAE (passengers)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Mean absolute error by demand stratum", fontsize=9)
    fig.tight_layout(pad=0.6)
    save(fig, "C5_mae_by_stratum.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# C6 / C7 — Scatter predito x real (um por modelo)
# ─────────────────────────────────────────────────────────────────────────────

def _plot_scatter_one(preds: np.ndarray, targets: np.ndarray,
                       cfg, name: str, fname: str) -> None:
    """
    Scatter com KDE, linha identidade e metricas no titulo.
    Identico ao plot_scatter de plot_results.py, mas para um modelo isolado.
    """
    cmap     = matplotlib.colormaps["viridis"]
    horizons = cfg.horizons_min

    all_true_flat = np.concatenate([
        targets[:, h, :].ravel()[
            np.isfinite(targets[:, h, :].ravel()) & (targets[:, h, :].ravel() > 0)
        ]
        for h in range(cfg.out_steps)
    ])
    lim_min = 0.0
    lim_max = float(np.percentile(all_true_flat, 99) * 1.05)

    rng      = np.random.default_rng(42)
    z_all, samples = [], []
    for h in range(cfg.out_steps):
        yt = targets[:, h, :].ravel()
        yp = preds[:, h, :].ravel()
        v  = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
        yt, yp = yt[v], yp[v]
        idx = rng.choice(len(yt), size=min(8000, len(yt)), replace=False)
        yt_s, yp_s = yt[idx], yp[idx]
        try:
            z = gaussian_kde(np.vstack([yt_s, yp_s]),
                             bw_method=0.15)(np.vstack([yt_s, yp_s]))
        except Exception:
            z = np.ones(len(yt_s))
        z_all.append(z)
        samples.append((yt_s, yp_s))

    z_min = min(z.min() for z in z_all)
    z_max = max(z.max() for z in z_all)
    norm  = mcolors.Normalize(vmin=z_min, vmax=z_max)

    ncols = min(cfg.out_steps, 3)
    nrows = math.ceil(cfg.out_steps / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(COL2 * ncols / 3 + 0.6, COL2 * 0.35 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for h in range(cfg.out_steps):
        ax         = axes_flat[h]
        yt_s, yp_s = samples[h]
        o          = z_all[h].argsort()
        ax.scatter(yt_s[o], yp_s[o], c=z_all[h][o], s=2, cmap=cmap,
                   norm=norm, rasterized=True, linewidths=0)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=0.8)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal")

        m   = compute_metrics(preds[:, h, :], targets[:, h, :])
        ytf = targets[:, h, :].ravel()
        ypf = preds[:, h, :].ravel()
        vf  = np.isfinite(ytf) & np.isfinite(ypf) & (ytf > 0)
        r2  = float(np.corrcoef(ytf[vf], ypf[vf])[0, 1] ** 2)
        ax.set_title(f"t+{horizons[h]} min  |  MAE={m['MAE']:.2f}  R²={r2:.3f}",
                     fontsize=8)
        ax.set_xlabel("Observed (passengers)")
        ax.set_ylabel("Predicted (passengers)")

    for h in range(cfg.out_steps, nrows * ncols):
        axes_flat[h].set_visible(False)

    fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.35)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.025, 0.70])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Point density (KDE)", fontsize=7, labelpad=6)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks([z_min, (z_min + z_max) / 2, z_max])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    fig.suptitle(name, fontsize=9, y=1.01, fontweight="bold")

    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    log.info(f"Salvo: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# C8 — CDF do erro absoluto (ambos os modelos, ambos os horizontes)
# ─────────────────────────────────────────────────────────────────────────────

def plot_cdf_comparison(results: dict, cfg) -> None:
    """
    Uma curva por (modelo, horizonte). Linhas solidas = AGCRN, tracejadas = GWN.
    Cores por horizonte, estilo por modelo.
    """
    horizons    = cfg.horizons_min
    h_colors    = [COLORS["blue"], COLORS["orange"]]
    model_styles = {"AGCRN": "-", "Graph Wavenet": "--"}

    fig, ax = plt.subplots(figsize=(COL2 * 0.65, COL2 * 0.42))
    x_max   = 0.0

    for name, res in results.items():
        for h in range(cfg.out_steps):
            err = np.abs(res["preds"][:, h, :] - res["targets"][:, h, :]).ravel()
            err = err[np.isfinite(err)]
            sorted_err = np.sort(err)
            cdf        = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
            step       = max(1, len(sorted_err) // 2000)
            ax.plot(sorted_err[::step], cdf[::step],
                    color=h_colors[h],
                    linestyle=model_styles[name],
                    linewidth=1.0,
                    label=f"{name}  t+{horizons[h]} min")
            x_max = max(x_max, float(sorted_err[-1]))

    ax.set_xlim(left=0, right=min(x_max, 150) * 1.02)
    ax.set_ylim(0, 1)
    ax.axhline(0.90, color="gray", linewidth=0.7, linestyle=":", alpha=0.8)
    ax.text(1.5, 0.915, "90th pct.", fontsize=7, color="gray")
    ax.set_xlabel("Absolute error (passengers)")
    ax.set_ylabel("Cumulative probability")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.set_title("CDF of absolute error — AGCRN vs. Graph WaveNet", fontsize=9)
    fig.tight_layout(pad=0.6)
    save(fig, "C8_error_cdf.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# C9 — Boxplot do erro absoluto por modelo e horizonte
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplot_comparison(results: dict, cfg) -> None:
    """
    Boxplot agrupado: pares (modelo, horizonte) no eixo x.
    Whiskers 5–95 percentil; sem outliers; mediana em preto.
    """
    horizons = cfg.horizons_min
    names    = list(results.keys())

    # Monta grupos: [AGCRN t+20, AGCRN t+40, GWN t+20, GWN t+40]
    group_data   = []
    group_labels = []
    group_colors = []

    for name in names:
        for h, hz in enumerate(horizons):
            err = np.abs(
                results[name]["preds"][:, h, :] - results[name]["targets"][:, h, :]
            ).ravel()
            group_data.append(err[np.isfinite(err)])
            group_labels.append(f"{name}\nt+{hz} min")
            group_colors.append(MODEL_COLORS[name])

    p97 = max(float(np.percentile(g, 97)) for g in group_data)

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.40))
    import matplotlib as _mpl_mod
    _mpl_ver      = tuple(int(x) for x in _mpl_mod.__version__.split(".")[:2])
    _bp_label_key = "tick_labels" if _mpl_ver >= (3, 9) else "labels"
    bp = ax.boxplot(
        group_data,
        **{_bp_label_key: group_labels},
        patch_artist= True,
        whis        = (5, 95),
        showfliers  = False,
        medianprops = dict(color="black", linewidth=1.2),
        whiskerprops= dict(linewidth=0.8),
        capprops    = dict(linewidth=0.8),
        widths      = 0.55,
    )
    for patch, col in zip(bp["boxes"], group_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    # Mediana anotada
    for i, (med_line, col) in enumerate(zip(bp["medians"], group_colors)):
        med = med_line.get_ydata()[0]
        ax.text(i + 1, med + p97 * 0.015, f"{med:.1f}",
                ha="center", va="bottom", fontsize=6,
                color=col, fontweight="bold")

    ax.set_ylim(0, p97 * 1.12)
    ax.set_xlabel("Model and forecast horizon")
    ax.set_ylabel("Absolute error (passengers)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.text(0.99, 0.97, "Whiskers: 5th–95th percentile",
            transform=ax.transAxes, fontsize=6, color="gray",
            ha="right", va="top")

    # Separador visual entre modelos
    ax.axvline(len(horizons) + 0.5, color="gray",
               linewidth=0.8, linestyle=":", alpha=0.7)

    fig.tight_layout(pad=0.6)
    save(fig, "C9_error_boxplot.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# C10 — Reducao relativa de MAE por estrato
# ─────────────────────────────────────────────────────────────────────────────

def plot_mae_gain(results: dict, cfg) -> None:
    """
    Barras horizontais: reducao percentual de MAE do AGCRN em relacao ao GWN,
    por horizonte e por estrato de demanda.
    Valores positivos = AGCRN melhor.
    """
    if "AGCRN" not in results or "Graph Wavenet" not in results:
        log.warning("C10: requer ambos AGCRN e GraphWaveNet — pulando.")
        return

    horizons = cfg.horizons_min
    ref_t    = results["AGCRN"]["targets"]
    strata   = {"Global": None, **stratify_stops(ref_t)}

    # Para cada horizonte e estrato, calcula a reducao
    rows = []
    for h, hz in enumerate(horizons):
        for group, idx in strata.items():
            if idx is not None and len(idx) == 0:
                continue
            sl_ag = (results["AGCRN"]["preds"][:, h, idx]
                     if idx is not None else results["AGCRN"]["preds"][:, h, :])
            tg_ag = (results["AGCRN"]["targets"][:, h, idx]
                     if idx is not None else results["AGCRN"]["targets"][:, h, :])
            sl_gw = (results["Graph Wavenet"]["preds"][:, h, idx]
                     if idx is not None else results["Graph Wavenet"]["preds"][:, h, :])
            tg_gw = (results["Graph Wavenet"]["targets"][:, h, idx]
                     if idx is not None else results["Graph Wavenet"]["targets"][:, h, :])

            mae_ag = float(np.nanmean(np.abs(sl_ag - tg_ag)))
            mae_gw = float(np.nanmean(np.abs(sl_gw - tg_gw)))
            gain   = (mae_gw - mae_ag) / mae_gw * 100 if mae_gw > 0 else 0.0
            rows.append({
                "horizon": f"t+{hz} min",
                "stratum": group.replace("\n", " "),
                "gain":    gain,
                "mae_ag":  mae_ag,
                "mae_gw":  mae_gw,
            })

    df = pd.DataFrame(rows)
    strata_order = ["Global"] + [s.replace("\n", " ") for s in list(stratify_stops(ref_t).keys())]
    df["stratum"] = pd.Categorical(df["stratum"], categories=strata_order, ordered=True)
    df = df.sort_values(["horizon", "stratum"])

    n_groups  = len(strata_order)
    n_horiz   = len(horizons)
    bar_h     = 0.32
    y_spacing = n_horiz * bar_h + 0.35

    fig, ax = plt.subplots(figsize=(COL2 * 0.70, n_groups * y_spacing * 0.55 + 0.6))

    h_colors = [COLORS["blue"], COLORS["orange"]]
    ytick_pos, ytick_labels = [], []

    for gi, stratum in enumerate(strata_order):
        sub   = df[df["stratum"] == stratum]
        y_ctr = gi * y_spacing
        for hi, (_, row) in enumerate(sub.iterrows()):
            y    = y_ctr + (hi - (n_horiz - 1) / 2) * bar_h
            col  = h_colors[hi % len(h_colors)]
            gain = row["gain"]
            ax.barh(y, gain, height=bar_h * 0.85,
                    color=col if gain >= 0 else COLORS["red"],
                    alpha=0.82, zorder=3)
            label = f"{gain:+.1f}%  ({row['mae_ag']:.2f} vs {row['mae_gw']:.2f})"
            ax.text(gain + (0.3 if gain >= 0 else -0.3), y,
                    label, va="center",
                    ha="left" if gain >= 0 else "right",
                    fontsize=5.8,
                    color=col if gain >= 0 else COLORS["red"])
            # Legenda de horizonte na primeira iteracao de estrato
            if gi == 0:
                ax.plot([], [], color=col, linewidth=4, alpha=0.75,
                        label=row["horizon"])
        ytick_pos.append(y_ctr)
        ytick_labels.append(stratum)

    ax.axvline(0, color="black", linewidth=0.8, zorder=4)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_xlabel("MAE reduction of AGCRN vs. Graph WaveNet (%)")
    ax.set_title("AGCRN vs. Graph WaveNet — MAE gain by stratum\n"
                 "(positive = AGCRN better)", fontsize=9)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.legend(title="Horizon", loc="lower right", fontsize=7)
    ax.invert_yaxis()
    fig.tight_layout(pad=0.6)
    save(fig, "C10_mae_gain.pdf")



# ─────────────────────────────────────────────────────────────────────────────
# C11 — Serie temporal comparativa por parada (AGCRN vs GWN)
# ─────────────────────────────────────────────────────────────────────────────

def _build_artifact_mask(real: np.ndarray, freq_min: int) -> np.ndarray:
    """
    Detecta trechos de imputacao via variancia local em janela deslizante
    de tamanho in_steps (8 para freq_min=20). Threshold=1e-3.
    Retorna bool array (n_steps,) — True onde e artefato.
    """
    window        = max(4, 160 // freq_min)
    VAR_THRESHOLD = 1e-3
    n             = len(real)
    artifact      = np.zeros(n, dtype=bool)
    for i in range(n):
        i0 = max(0, i - window)
        i1 = min(n, i + window + 1)
        if real[i0:i1].var() < VAR_THRESHOLD:
            artifact[i] = True
    return artifact


def _shade_artifact_regions(ax, time_axis: np.ndarray,
                              artifact: np.ndarray,
                              real: np.ndarray,
                              label_first: bool = False) -> None:
    """
    Sombreia em cinza todos os intervalos contiguos onde artifact=True.
    Insere label "Imputed data" no primeiro intervalo se label_first=True.
    """
    padded = np.concatenate([[False], artifact, [False]])
    starts = np.where(~padded[:-1] &  padded[1:])[0]
    ends   = np.where( padded[:-1] & ~padded[1:])[0]
    first_labeled = False
    for s, e in zip(starts, ends):
        t_s = time_axis[s]
        t_e = time_axis[min(e, len(time_axis) - 1)]
        ax.axvspan(t_s, t_e, color="gray", alpha=0.13, zorder=5, linewidth=0)
        ax.axvline(t_s, color="gray", linewidth=0.5, linestyle="--", alpha=0.6, zorder=6)
        ax.axvline(t_e, color="gray", linewidth=0.5, linestyle="--", alpha=0.6, zorder=6)
        if label_first and not first_labeled:
            mid   = (t_s + t_e) / 2
            y_pos = real[s:e].mean() if e > s else real[s]
            ax.text(mid, y_pos, "Imputed data",
                    fontsize=8, color="gray", ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            first_labeled = True


def _node_label_cmp(local_idx: int, global_indices: np.ndarray, node_meta) -> str:
    global_idx = int(global_indices[local_idx])
    if node_meta is None or global_idx not in node_meta.index:
        return f"Active node #{global_idx}"
    row       = node_meta.loc[global_idx]
    stop_id   = row.get("stop_id",          "?")
    route     = row.get("route_short_name", "?")
    direction = row.get("direction_id",     "?")
    dir_label = {0: "Inbound", 1: "Outbound"}.get(int(direction), str(direction))
    return f"Stop {stop_id} | Route {route} | {dir_label}  (node idx: {global_idx})"


def _plot_one_stop_model(preds: np.ndarray, targets: np.ndarray,
                          cfg, model_name: str, pred_color: str,
                          local_idx: int, fname: str,
                          global_indices: np.ndarray,
                          node_meta, n_steps: int = 96) -> None:
    """
    Replica exata de _plot_one_stop do plot_results.py para um unico modelo.

    Gera cfg.out_steps subplots empilhados (um por horizonte). Em cada subplot:
      - Linha observada (preto solido)
      - Predicao do modelo (pred_color, tracejado)
      - Fill entre observado e predito (alpha=0.15)
      - Regioes de imputacao sombreadas em cinza
      - MAE calculado apenas sobre timesteps reais (artifact=False)
      - Percentual de dados reais anotado no titulo do subplot

    Args:
        preds, targets : (T, out_steps, N_active) — escala real, pos-filtro
        model_name     : string exibido no suptitle e nas legendas
        pred_color     : cor da linha de predicao
    """
    horizons  = cfg.horizons_min
    time_axis = np.arange(n_steps) * cfg.freq_min

    # Cores por horizonte — identico ao plot_results.py
    hcolors = [COLORS["blue"], COLORS["sky"], COLORS["green"],
               COLORS["orange"], COLORS["purple"], COLORS["red"]]

    if cfg.out_steps == 1:
        fig, axes_arr = plt.subplots(1, 1, figsize=(COL2, COL2 * 0.28))
        axes_list = [axes_arr]
    else:
        fig, axes_arr = plt.subplots(
            cfg.out_steps, 1,
            figsize=(COL2, COL2 * 0.22 * cfg.out_steps),
            sharex=True,
        )
        axes_list = list(axes_arr)

    real_full = targets[:n_steps, 0, local_idx]
    artifact  = _build_artifact_mask(real_full, cfg.freq_min)
    real_mask = ~artifact
    n_real    = int(real_mask.sum())
    n_total   = len(real_full)
    pct_real  = n_real / max(n_total, 1) * 100
    log.info(
        f"  C11 [{model_name}] local_idx={local_idx}: "
        f"{n_real}/{n_total} timesteps reais ({pct_real:.1f}%)"
    )

    for h, (ax, hz, col) in enumerate(zip(axes_list, horizons, hcolors)):
        real = targets[:n_steps, 0, local_idx]
        pred = preds[:n_steps, h, local_idx]

        ax.plot(time_axis, real, color=COLORS["black"], linewidth=0.85,
                label="Observed", zorder=4)
        ax.plot(time_axis, pred, color=pred_color, linewidth=0.9, linestyle="--",
                label=f"Predicted (t+{hz} min)", zorder=3)
        ax.fill_between(time_axis, real, pred, alpha=0.15, color=pred_color, zorder=2)

        if real_mask.any():
            mae_local = float(np.nanmean(np.abs(pred[real_mask] - real[real_mask])))
            mae_label = f"{mae_local:.2f} pass."
        else:
            mae_label = "n/a (without real data)"

        ax.text(0.01, 0.97,
                f"t+{hz} min  |  MAE = {mae_label}  [{pct_real:.0f}% real data]",
                transform=ax.transAxes, fontsize=10, va="top")

        # Regioes de imputacao sombreadas apenas no subplot t+20 (h==0)
        _shade_artifact_regions(ax, time_axis, artifact, real, label_first=(h == 0))

        ax.set_ylabel("Loading\n(pass.)")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        # Legenda em todos os subplots — label da predicao atualizado por horizonte
        ax.legend(loc="upper right", ncol=2, handlelength=1.4, fontsize=10)

    axes_list[-1].set_xlabel("Time (min)")
    axes_list[-1].xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.suptitle(
        f"{_node_label_cmp(local_idx, global_indices, node_meta)}"
        f" -- all forecast horizons  [{model_name}]",
        fontsize=11, y=1.005,
    )
    fig.tight_layout(pad=0.4)
    save(fig, fname)


def plot_temporal_comparison(results: dict, cfg, node_meta,
                              global_indices: np.ndarray,
                              n_stops: int = 3,
                              n_steps: int = 96) -> None:
    """
    Para cada uma das n_stops paradas de maior demanda media, gera uma figura
    separada por modelo (AGCRN e GWN), identica ao formato do plot_results.py.

    Saida por parada (ex.: stop 1):
      C11a_agcrn_stop1.pdf        — AGCRN, parada 1
      C11a_gwn_stop1.pdf          — GWN, parada 1
      C11b_agcrn_stop2.pdf / ...  — paradas 2 e 3

    As paradas sao selecionadas pelos maiores valores de demanda media nos
    targets (identico ao criterio de plot_temporal_forecast em plot_results.py).
    O indice de parada e compartilhado entre modelos — ambas as figuras mostram
    exatamente a mesma parada para comparacao direta.
    """
    model_colors = {
        "AGCRN":        COLORS["orange"],
        "Graph Wavenet": COLORS["blue"],
    }
    model_slugs = {
        "AGCRN":        "agcrn",
        "Graph Wavenet": "gwn",
    }
    stop_letters = ["a", "b", "c", "d", "e"]

    ref_targets = list(results.values())[0]["targets"]
    with np.errstate(all="ignore"):
        mean_demand = np.nanmean(ref_targets, axis=(0, 1))

    valid_local = np.where(np.isfinite(mean_demand))[0]
    top_local   = valid_local[
        np.argsort(mean_demand[valid_local])[-n_stops:][::-1]
    ]

    for si, local_idx in enumerate(top_local):
        letter = stop_letters[si] if si < len(stop_letters) else str(si + 1)
        for name, res in results.items():
            slug  = model_slugs.get(name, name.lower().replace(" ", "_"))
            color = model_colors.get(name, COLORS["green"])
            fname = f"C11{letter}_{slug}_stop{si+1}.pdf"
            _plot_one_stop_model(
                preds         = res["preds"],
                targets       = res["targets"],
                cfg           = cfg,
                model_name    = name,
                pred_color    = color,
                local_idx     = int(local_idx),
                fname         = fname,
                global_indices= global_indices,
                node_meta     = node_meta,
                n_steps       = n_steps,
            )




# ─────────────────────────────────────────────────────────────────────────────
# C12 — Tabela de metricas por horizonte e estrato
# ─────────────────────────────────────────────────────────────────────────────

# Nomes limpos dos estratos para exibicao (sem quebra de linha)
_STRATA_DISPLAY = {
    "Low\n(<5)":      "Low (<5)",
    "Medium\n(5–20)": "Medium (5–20)",
    "High\n(≥20)":    "High (≥20)",
}

# Metricas exibidas e seus rotulos
_METRICS = [
    ("MAE",       "MAE"),
    ("RMSE",      "RMSE"),
    ("MAPE_filt", "MAPE"),
    ("sMAPE",     "sMAPE"),
]


def _build_table_data(results: dict, cfg) -> tuple:
    """
    Constroi todos os valores da tabela de uma vez.

    Retorna (df, horizons, strata_names, model_names) onde df e um
    DataFrame com MultiIndex (horizonte, estrato) e colunas
    (model, metric) — pronto para pivot e formatacao.

    Estrutura final de df:
        horizonte | estrato   | model        | MAE  | RMSE | MAPE | sMAPE
        Global    | Global    | AGCRN        | ...
        Global    | Global    | GraphWaveNet | ...
        Global    | Low (<5)  | AGCRN        | ...
        ...
        t+20 min  | Global    | AGCRN        | ...
        ...
    """
    ref_targets  = list(results.values())[0]["targets"]
    strata_raw   = stratify_stops(ref_targets)
    strata_items = {"Global": None, **strata_raw}  # Global primeiro
    horizons_min = cfg.horizons_min
    model_names  = list(results.keys())

    rows = []
    for name, res in results.items():
        p_all = res["preds"]   # (T, out_steps, N)
        t_all = res["targets"] # (T, out_steps, N)

        horizon_specs = [("Global", None)] + [
            (f"t+{hz} min", h) for h, hz in enumerate(horizons_min)
        ]

        for label_h, h_idx in horizon_specs:
            # h_idx=None  -> todos os horizontes, p_h shape (T, out_steps, N)
            # h_idx=int   -> horizonte especifico,  p_h shape (T, N)
            if h_idx is None:
                p_h = p_all          # (T, out_steps, N)
                t_h = t_all
            else:
                p_h = p_all[:, h_idx, :]   # (T, N)
                t_h = t_all[:, h_idx, :]

            for strata_key, idx in strata_items.items():
                strata_label = _STRATA_DISPLAY.get(strata_key, strata_key)

                # Filtra por estrato no eixo N (ultimo eixo)
                if idx is not None:
                    p_s = p_h[..., idx]    # (T, out_steps, n_s) ou (T, n_s)
                    t_s = t_h[..., idx]
                else:
                    p_s = p_h
                    t_s = t_h

                # compute_metrics espera arrays 2D ou "achatados"
                # Achata (T, out_steps, N) -> (T*out_steps, N) para calcular
                # Achata (T, N)            -> ja e 2D, sem alteracao
                if p_s.ndim == 3:
                    T, S, N_ = p_s.shape
                    p_s = p_s.reshape(T * S, N_)
                    t_s = t_s.reshape(T * S, N_)

                m = compute_metrics(p_s, t_s)
                rows.append({
                    "Horizon":  label_h,
                    "Stratum":  strata_label,
                    "Model":    name,
                    **{k: m[key] for key, k in _METRICS},
                })

    df = pd.DataFrame(rows)

    horizon_order = ["Global"] + [f"t+{hz} min" for hz in horizons_min]
    strata_order  = ["Global"] + [
        _STRATA_DISPLAY.get(k, k) for k in strata_raw.keys()
    ]
    df["Horizon"] = pd.Categorical(df["Horizon"], categories=horizon_order, ordered=True)
    df["Stratum"] = pd.Categorical(df["Stratum"], categories=strata_order,  ordered=True)
    df = df.sort_values(["Horizon", "Stratum", "Model"]).reset_index(drop=True)

    return df, horizon_order, strata_order, model_names


def print_metrics_table(results: dict, cfg) -> None:
    """
    Exibe a tabela completa no console via logging.

    Formato:
      - Uma secao por horizonte temporal
      - Dentro de cada secao: linhas = estratos, colunas = (modelo, metrica)
      - Separadores de secao em ASCII para legibilidade no terminal
    """
    df, horizon_order, strata_order, model_names = _build_table_data(results, cfg)

    metric_keys = [k for _, k in _METRICS]
    col_w       = 9    # largura de cada coluna de metrica
    strat_w     = 16   # largura da coluna de estrato

    # Cabecalho de modelos e metricas
    def _make_header():
        line1 = f"  {'':<{strat_w}}"
        line2 = f"  {'':<{strat_w}}"
        for name in model_names:
            short = name[:16]
            n_cols = len(metric_keys)
            block_w = col_w * n_cols + n_cols - 1
            line1 += f"  {short:^{block_w}}"
            line2 += "  " + "  ".join(f"{k:>{col_w}}" for k in metric_keys)
        return line1, line2

    SEP_WIDE = "=" * (strat_w + (col_w * len(metric_keys) + len(metric_keys)) * len(model_names) + 6)
    SEP_THIN = "-" * len(SEP_WIDE)

    log.info("")
    log.info(SEP_WIDE)
    log.info("  METRICS TABLE — by horizon and demand stratum")
    log.info(f"  Metrics: MAE / RMSE / MAPE (>2 pass.) / sMAPE  |  freq_min={cfg.freq_min} min")
    log.info(SEP_WIDE)

    for hz in horizon_order:
        log.info("")
        log.info(f"  Horizon: {hz}")
        log.info(SEP_THIN)
        line1, line2 = _make_header()
        log.info(line1)
        log.info(line2)
        log.info(SEP_THIN)

        sub = df[df["Horizon"] == hz]
        for stratum in strata_order:
            row_str = f"  {stratum:<{strat_w}}"
            for name in model_names:
                cell = sub[(sub["Stratum"] == stratum) & (sub["Model"] == name)]
                if cell.empty:
                    row_str += "  " + "  ".join(f"{'n/a':>{col_w}}" for _ in metric_keys)
                else:
                    r = cell.iloc[0]
                    for k in metric_keys:
                        v = r[k]
                        row_str += f"  {v:>{col_w}.2f}" if not np.isnan(v) else f"  {'n/a':>{col_w}}"
            log.info(row_str)
        log.info(SEP_THIN)

    log.info("")


def plot_metrics_table(results: dict, cfg) -> None:
    """
    Renderiza a tabela como figura matplotlib e salva em PDF + PNG.

    Layout: uma sub-tabela por horizonte temporal, empilhadas verticalmente.
    Colunas: Stratum | (AGCRN: MAE RMSE MAPE sMAPE) | (GWN: MAE RMSE MAPE sMAPE)
    Cores:
      - Cabecalhos de secao em cinza escuro
      - Cabecalhos de modelo nas cores do modelo (laranja / azul)
      - Linhas alternadas em branco / cinza claro
      - Melhor valor em cada linha em negrito verde
      - Global em fundo levemente destacado
    """
    df, horizon_order, strata_order, model_names = _build_table_data(results, cfg)

    metric_keys   = [k for _, k in _METRICS]
    metric_labels = [lbl for _, lbl in _METRICS]
    n_models      = len(model_names)
    n_metrics     = len(metric_keys)
    n_strata      = len(strata_order)

    # Larguras relativas das colunas: [estrato] + [metrica x modelo x n_metrics]
    col_widths = [2.2] + [1.1] * (n_models * n_metrics)
    n_cols     = len(col_widths)

    # Cores
    clr_model  = {
        "AGCRN":        MODEL_COLORS["AGCRN"],
        "Graph Wavenet": MODEL_COLORS["Graph Wavenet"],
    }
    clr_header_bg  = "#2d2d2d"
    clr_header_txt = "white"
    clr_row_odd    = "#f5f5f5"
    clr_row_even   = "white"
    clr_global_bg  = "#eef4fb"
    clr_best       = "#1a7a3c"   # verde escuro para melhor valor
    clr_sep        = "#cccccc"

    n_horizons = len(horizon_order)
    # Altura: cabecalho geral (2 linhas) + por horizonte (cabecalho 2 linhas + n_strata linhas)
    rows_per_hz = 2 + n_strata
    total_rows  = 2 + n_horizons * rows_per_hz

    row_h  = 0.32   # polegadas por linha
    fig_h  = total_rows * row_h + 0.5
    fig_w  = sum(col_widths) + 0.3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, sum(col_widths))
    ax.set_ylim(0, fig_h - 0.5)
    ax.axis("off")

    def _cell(x, y, w, text, bg, fg="black", bold=False,
               fontsize=6.5, ha="center", va="center"):
        ax.add_patch(plt.Rectangle((x, y), w, row_h, color=bg,
                                    linewidth=0, zorder=1))
        ax.text(x + w * (0.08 if ha == "left" else 0.5),
                y + row_h * 0.5, text,
                color=fg, fontsize=fontsize,
                fontweight="bold" if bold else "normal",
                ha=ha, va=va, zorder=2, clip_on=True)

    def _hline(y, color=clr_sep, lw=0.5):
        ax.plot([0, sum(col_widths)], [y, y], color=color, lw=lw, zorder=3)

    cur_y = fig_h - 0.5 - row_h   # y decresce de cima para baixo

    # ── Cabecalho geral — linha 1: modelos ────────────────────────────────────
    x = 0
    _cell(x, cur_y, col_widths[0], "", clr_header_bg, clr_header_txt)
    x += col_widths[0]
    for name in model_names:
        block_w = sum(col_widths[1:1 + n_metrics])
        ax.add_patch(plt.Rectangle((x, cur_y), block_w, row_h,
                                    color=clr_model.get(name, "#555"),
                                    linewidth=0, zorder=1))
        ax.text(x + block_w / 2, cur_y + row_h * 0.5,
                name, color="white", fontsize=7, fontweight="bold",
                ha="center", va="center", zorder=2)
        x += block_w
    _hline(cur_y)
    cur_y -= row_h

    # ── Cabecalho geral — linha 2: metricas ───────────────────────────────────
    x = 0
    _cell(x, cur_y, col_widths[0], "Stratum",
          clr_header_bg, clr_header_txt, bold=True, fontsize=6.5)
    x += col_widths[0]
    for _ in model_names:
        for lbl, cw in zip(metric_labels, col_widths[1:1 + n_metrics]):
            _cell(x, cur_y, cw, lbl, clr_header_bg, clr_header_txt,
                  bold=True, fontsize=6.5)
            x += cw
    _hline(cur_y)
    _hline(cur_y + row_h)
    cur_y -= row_h

    # ── Secao por horizonte ───────────────────────────────────────────────────
    for hz in horizon_order:
        sub = df[df["Horizon"] == hz]

        # Titulo do horizonte
        _hline(cur_y + row_h, color="#888", lw=0.8)
        ax.add_patch(plt.Rectangle((0, cur_y), sum(col_widths), row_h,
                                    color="#444444", linewidth=0, zorder=1))
        ax.text(0.08, cur_y + row_h * 0.5,
                f"Horizon: {hz}",
                color="white", fontsize=7, fontweight="bold",
                ha="left", va="center", zorder=2)
        _hline(cur_y)
        cur_y -= row_h

        # Sub-cabecalho de metricas (repetido por secao)
        x = 0
        _cell(x, cur_y, col_widths[0], "Stratum",
              "#555555", "white", bold=True, fontsize=6)
        x += col_widths[0]
        for name in model_names:
            for lbl, cw in zip(metric_labels, col_widths[1:1 + n_metrics]):
                _cell(x, cur_y, cw, lbl, "#555555", "white",
                      bold=True, fontsize=6)
                x += cw
        _hline(cur_y)
        cur_y -= row_h

        # Linhas de dados
        for si, stratum in enumerate(strata_order):
            is_global = stratum == "Global"
            bg_row    = clr_global_bg if is_global else (
                clr_row_odd if si % 2 == 0 else clr_row_even
            )

            # Determina melhor valor por metrica nesta linha (menor = melhor)
            best_vals = {}
            for k in metric_keys:
                vals = []
                for name in model_names:
                    cell = sub[(sub["Stratum"] == stratum) & (sub["Model"] == name)]
                    if not cell.empty:
                        v = cell.iloc[0][k]
                        if not np.isnan(v):
                            vals.append(v)
                if vals:
                    best_vals[k] = min(vals)

            x = 0
            _cell(x, cur_y, col_widths[0], stratum, bg_row,
                  ha="left", fontsize=6.5, bold=is_global)
            x += col_widths[0]

            for name in model_names:
                cell = sub[(sub["Stratum"] == stratum) & (sub["Model"] == name)]
                for k, cw in zip(metric_keys, col_widths[1:1 + n_metrics]):
                    if cell.empty:
                        _cell(x, cur_y, cw, "n/a", bg_row, fontsize=6)
                    else:
                        v = cell.iloc[0][k]
                        if np.isnan(v):
                            _cell(x, cur_y, cw, "n/a", bg_row, fontsize=6)
                        else:
                            is_best = k in best_vals and abs(v - best_vals[k]) < 1e-6
                            _cell(x, cur_y, cw, f"{v:.2f}", bg_row,
                                  fg=clr_best if is_best else "black",
                                  bold=is_best, fontsize=6.5)
                    x += cw

            _hline(cur_y, lw=0.3)
            cur_y -= row_h

        _hline(cur_y + row_h, color="#aaa", lw=0.6)

    # Borda externa
    for side in [[0, 0, 0, fig_h - 0.5],
                 [sum(col_widths), 0, sum(col_widths), fig_h - 0.5],
                 [0, 0, sum(col_widths), 0],
                 [0, fig_h - 0.5, sum(col_widths), fig_h - 0.5]]:
        ax.plot(side[:2], side[2:], color="#555", lw=0.8, zorder=4)

    # Nota de rodape
    ax.text(0.02, -0.05,
            "Bold green = best value per row (lower is better). "
            "MAPE filtered for demand > 2 pass./window.",
            transform=ax.transAxes, fontsize=5.5, color="gray", va="top")

    fig.tight_layout(pad=0.1)
    save(fig, "C12_metrics_table.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== Comparacao AGCRN vs GraphWaveNet — Geracao de Figuras ===")

    results, cfg, active_mask = load_all_predictions()

    names    = list(results.keys())
    horizons = cfg.horizons_min
    log.info(f"Modelos carregados: {names}")
    log.info(f"Horizontes: {horizons} | out_steps: {cfg.out_steps}")
    log.info(f"Shape preds AGCRN: {results['AGCRN']['preds'].shape}")

    # node_meta e global_indices necessarios para C11 (rotulo das paradas)
    _ref_targets    = results[names[0]]["targets"]
    _mean_dem       = np.nanmean(_ref_targets, axis=(0, 1))
    global_indices  = np.where(np.isfinite(_mean_dem))[0]
    _meta_path      = os.path.join(cfg.ARTIFACTS_DIR, "node_metadata.csv")
    node_meta       = (
        pd.read_csv(_meta_path).set_index("node_index")
        if os.path.exists(_meta_path) else None
    )

    log.info("Gerando C1–C3: metricas por horizonte...")
    plot_metric_per_horizon(results, cfg, "MAE",    "MAE (passengers)",  "C1_mae_per_horizon.pdf")
    plot_metric_per_horizon(results, cfg, "RMSE",   "RMSE (passengers)", "C2_rmse_per_horizon.pdf")
    plot_metric_per_horizon(results, cfg, "sMAPE",  "sMAPE (%)",         "C3_smape_per_horizon.pdf")

    log.info("Gerando C4: radar chart...")
    plot_radar(results, cfg)

    log.info("Gerando C5: MAE por estrato...")
    plot_mae_by_stratum(results)

    log.info("Gerando C6/C7: scatter por modelo...")
    for name, fname in [("AGCRN", "C6_scatter_agcrn.pdf"),
                         ("Graph Wavenet", "C7_scatter_gwn.pdf")]:
        _plot_scatter_one(
            results[name]["preds"], results[name]["targets"],
            cfg, name, fname
        )

    log.info("Gerando C8: CDF comparativa...")
    plot_cdf_comparison(results, cfg)

    log.info("Gerando C9: boxplot comparativo...")
    plot_boxplot_comparison(results, cfg)

    log.info("Gerando C10: ganho de MAE por estrato...")
    plot_mae_gain(results, cfg)

    log.info("Gerando C11: series temporais comparativas...")
    plot_temporal_comparison(results, cfg, node_meta, global_indices)

    log.info("Gerando C12: tabela de metricas por horizonte e estrato...")
    print_metrics_table(results, cfg)
    plot_metrics_table(results, cfg)

    log.info(f"Todas as figuras salvas em: {OUT_DIR}")