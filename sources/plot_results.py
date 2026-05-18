"""
plot_results.py — versao corrigida para pipeline v7 (janela 20 min)
====================================================================
Gera graficos de avaliacao do GraphWaveNet.
Metricas calculadas apenas sobre nos ativos (threshold percentilico 5%).

Saida (results/figures/):
    01_metrics_per_horizon.pdf
    02_scatter_pred_vs_real.pdf
    03_error_distribution.pdf
    04a_temporal_stop_1.pdf
    04b_temporal_stop_2.pdf
    04c_temporal_stop_3.pdf
    05_spatial_error_map.pdf
    06_cumulative_error.pdf

Correcoes vs. versao anterior:
  [P1] horizons = cfg.horizons_min em todas as funcoes de plot.
       Substitui (h+1)*5 hardcoded. Com freq_min=20: exibe [20, 40].

  [P2] time_axis = np.arange(n_steps) * cfg.freq_min.
       Eixo temporal da serie temporal por parada em multiplos de
       freq_min (20 min). Com n_steps=96: mostra 96x20=1920 min (32h).

  [P3] n_steps adaptado ao freq_min do pipeline.
       O significado de n_steps mudou corretamente com time_axis.
       Para exibir um dia completo (1440 min), use n_steps=1440//freq_min=72.
       Mantido em 96 como padrao conservador; o chamador pode ajustar.

  [P4] plot_scatter corrigido: grid dinamico baseado em out_steps.
       Versao anterior criava subplots(2,3) fixo para out_steps=2,
       gerando 6 paineis mas apenas 2 com dados — os 4 restantes
       causavam IndexError em axes.flatten()[h] para h>=2.
       Agora: ncols=min(out_steps, 3), nrows=ceil(out_steps/ncols).
       Com out_steps=2: subplots(1,2) — exatamente 2 paineis.

  [P5] plot_cdf: label das curvas via cfg.horizons_min.
       Era f"t+{(h+1)*5} min" hardcoded.

  [P6] Validacao de alinhamento node_order vs stop_order (IC-P1 — CRITICO):
       Replica a verificacao [E4] de evaluate.py: compara node_order salvo
       no checkpoint (campo adicionado por train.py [T7]) com stop_order
       derivado de train_mean.csv. Divergencia levanta ValueError explicito.
       Fallback conservador (aviso) para checkpoints legados sem o campo.

  [P7] Validacao de in_features antes de construir o modelo (IC-P1 — CRITICO):
       Replica a verificacao [E5] de evaluate.py: assert X_test.shape[2] ==
       cfg.in_features e comparacao com o campo "in_features" do checkpoint.
       Previne shape mismatch opaco em load_state_dict.

  [P8] make_loaders chamado com X_train=None e X_val=None (IC-D2):
       plot_results.py usa apenas test_loader. A chamada anterior passava
       X_train e X_val inteiros e construia DataLoaders nunca iterados.
       Agora passa None; dataset.py [D1] suporta o contrato.

  [P9] Deteccao de artefatos de imputacao em toda a serie temporal (CORRECAO):
       A versao anterior detectava apenas o trecho inicial constante da serie
       (primeiro boundary) usando um loop que parava na primeira variacao > 2.0.
       Trechos constantes subsequentes — periodos sem operacao (feriados, grade
       noturna) imputados com zero normalizado (que desnormaliza para mu_n, ex:
       ~135 pass.) — apareciam como linha preta reta em regioes brancas do
       grafico, criando a falsa impressao de observacoes reais com valor constante.
       Correcao: nova funcao _build_artifact_mask() varre TODA a serie com janela
       deslizante de tamanho in_steps (8 timesteps = 160 min) e detecta qualquer
       trecho onde a variancia local e inferior a 1e-3 — seja no inicio, meio ou
       fim da janela plotada. _shade_artifact_regions() sombreia todos esses
       trechos em cinza e insere o label "Imputed data" no primeiro intervalo.
       MAE no titulo do painel calculado apenas sobre timesteps reais (nao
       imputados), evitando inflacao da metrica por periodos sem operacao.
"""

import os
import math
import logging

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import gaussian_kde

from config  import Config
from model   import GraphWaveNet
from dataset import make_loaders

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler("plot_results.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

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
C = list(COLORS.values())

COL1 = 3.45
COL2 = 7.16

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "figures"
)
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utilitarios
# ─────────────────────────────────────────────────────────────────────────────

def build_active_mask(X_test: np.ndarray, target_col_idx: int,
                      percentile: float = 5.0) -> np.ndarray:
    """
    Threshold percentilico (5%) sobre variancias positivas.
    Identico ao evaluate.py para consistencia de nos ativos entre scripts.
    """
    var_por_no   = X_test[:, :, target_col_idx].var(axis=0)
    var_positiva = var_por_no[var_por_no > 0]
    if len(var_positiva) == 0:
        log.warning("Todos os nos tem variancia zero.")
        return np.zeros(X_test.shape[1], dtype=bool)
    threshold = float(np.percentile(var_positiva, percentile))
    log.info(f"active_mask threshold (p{percentile}%): {threshold:.2e}")
    return var_por_no > threshold


def desnormalize(values, mean_series, std_series, stop_order):
    means = mean_series.reindex(stop_order, fill_value=0.0).values
    stds  = std_series.reindex(stop_order,  fill_value=1.0).values
    return values * stds[None, None, :] + means[None, None, :]


def compute_metrics(pred, target):
    with np.errstate(all="ignore"):
        mae  = float(np.nanmean(np.abs(pred - target)))
        rmse = float(np.sqrt(np.nanmean((pred - target) ** 2)))
        mask = target > 2
        mape = float(np.nanmean(
            np.abs((pred[mask] - target[mask]) / target[mask])
        ) * 100) if mask.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    fig.savefig(path.replace(".pdf", ".png"))
    log.info(f"Salvo: {path}")
    plt.close(fig)


def _node_label(local_idx, global_indices, node_meta):
    global_idx = int(global_indices[local_idx])
    if node_meta is None or global_idx not in node_meta.index:
        return f"Active node #{global_idx}"
    row       = node_meta.loc[global_idx]
    stop_id   = row.get("stop_id",          "?")
    route     = row.get("route_short_name", "?")
    direction = row.get("direction_id",     "?")
    dir_label = {0: "Inbound", 1: "Outbound"}.get(int(direction), str(direction))
    return f"Stop {stop_id} | Route {route} | {dir_label}  (node idx: {global_idx})"


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de predicoes
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def load_predictions():
    cfg         = Config()
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.type
    use_amp     = cfg.use_amp and device_type == "cuda"

    # [P1] horizons via cfg.horizons_min — com freq_min=20: [20, 40].
    log.info(f"freq_min={cfg.freq_min}min | horizons={cfg.horizons_min}")

    X_test   = np.load(os.path.join(cfg.ARTIFACTS_DIR, "X_test.npy"))
    adj_geo  = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_geo.npy"))
    adj_topo = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_topo.npy"))

    cfg.num_nodes = X_test.shape[1]
    N = cfg.num_nodes

    # [P7] Verifica in_features antes de construir o modelo (IC-P1 / replica [E5]).
    #      Previne shape mismatch opaco dentro de load_state_dict.
    assert X_test.shape[2] == cfg.in_features, (
        f"in_features no tensor de teste ({X_test.shape[2]}) != cfg ({cfg.in_features}). "
        f"Verifique ALL_FEATURES no pipeline e in_features em config.py."
    )

    active_mask    = build_active_mask(X_test, cfg.target_col_idx, percentile=5.0)
    n_active       = int(active_mask.sum())
    global_indices = np.where(active_mask)[0]
    log.info(f"Nos ativos: {n_active:,} / {N:,} ({(N-n_active)/N*100:.1f}% excluidos)")

    # [P8] X_train=None e X_val=None: apenas test_loader e necessario.
    #      dataset.py [D1] suporta splits None retornando None no lugar do loader.
    _, _, test_loader = make_loaders(None, None, X_test, cfg)

    model = GraphWaveNet(cfg, adj_geo, adj_topo).to(device)

    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state" in ckpt:

        # [P7] Valida in_features do checkpoint contra o tensor de teste.
        ckpt_in_features = ckpt.get("in_features")
        if ckpt_in_features is not None and int(ckpt_in_features) != cfg.in_features:
            raise ValueError(
                f"Checkpoint treinado com in_features={ckpt_in_features}, "
                f"mas cfg.in_features={cfg.in_features} e tensor tem F={X_test.shape[2]}. "
                f"Verifique consistencia entre pipeline, config.py e checkpoint."
            )

        model.load_state_dict(ckpt["model_state"])
        log.info(
            f"Checkpoint: epoca {ckpt.get('epoch','?')} | "
            f"val MAE: {ckpt.get('best_val', float('nan')):.4f} | "
            f"freq_min={ckpt.get('freq_min','?')}min"
        )

        # Aviso se o checkpoint foi treinado com freq_min diferente.
        ckpt_freq = ckpt.get("freq_min", "?")
        if ckpt_freq != "?" and int(ckpt_freq) != cfg.freq_min:
            log.warning(
                f"ATENCAO: checkpoint treinado com freq_min={ckpt_freq}min, "
                f"mas cfg.freq_min={cfg.freq_min}min. Verifique consistencia."
            )

    else:
        model.load_state_dict(ckpt)
        log.info("Checkpoint legado (state_dict direto).")

    model.eval()

    train_mean = pd.read_csv(os.path.join(cfg.ARTIFACTS_DIR, "train_mean.csv"), index_col=0)
    train_std  = pd.read_csv(os.path.join(cfg.ARTIFACTS_DIR, "train_std.csv"),  index_col=0)
    stop_order = list(train_mean.index)

    # [P6] Validacao de alinhamento node_order vs stop_order (IC-P1 / replica [E4]).
    #      stop_order (de train_mean.csv) deve ser identico a node_order usado no
    #      treinamento para que desnormalize() mapeie cada coluna do tensor ao no
    #      correto. Checkpoints gerados por train.py [T7] incluem "node_order".
    if isinstance(ckpt, dict):
        ckpt_node_order = ckpt.get("node_order")
        if ckpt_node_order is not None:
            if ckpt_node_order != stop_order:
                n_diff = sum(a != b for a, b in zip(ckpt_node_order, stop_order))
                raise ValueError(
                    f"Desalinhamento entre node_order do checkpoint e stop_order de "
                    f"train_mean.csv: {n_diff} posicoes divergem (total N={N}). "
                    f"Regenere os artefatos ou certifique-se de usar o mesmo pipeline."
                )
            else:
                log.info("Alinhamento node_order OK — checkpoint e train_mean.csv consistentes.")
        else:
            log.warning(
                "Checkpoint legado sem campo 'node_order' — validacao de alinhamento ignorada. "
                "Retreine com train.py atualizado para habilitar esta verificacao."
            )

    meta_path = os.path.join(cfg.ARTIFACTS_DIR, "node_metadata.csv")
    node_meta = pd.read_csv(meta_path).set_index("node_index") if os.path.exists(meta_path) else None

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

    preds_real   = desnormalize(preds,   train_mean["loading"], train_std["loading"], stop_order)
    targets_real = desnormalize(targets, train_mean["loading"], train_std["loading"], stop_order)

    preds_a   = preds_real[:, :, active_mask]
    targets_a = targets_real[:, :, active_mask]

    log.info(f"Inferencia concluida -- shape ativo: {preds_a.shape}")
    return preds_a, targets_a, cfg, active_mask, node_meta, global_indices


# ─────────────────────────────────────────────────────────────────────────────
# Figura 1 — Metricas por horizonte
# ─────────────────────────────────────────────────────────────────────────────

def plot_metrics_per_horizon(preds, targets, cfg):
    """[P1] horizons via cfg.horizons_min — exibe [20, 40] para freq_min=20."""
    horizons = cfg.horizons_min   # [20, 40]
    maes, rmses, mapes = [], [], []
    for h in range(cfg.out_steps):
        m = compute_metrics(preds[:, h, :], targets[:, h, :])
        maes.append(m["MAE"]); rmses.append(m["RMSE"]); mapes.append(m["MAPE"])

    fig, axes = plt.subplots(1, 3, figsize=(COL2, COL2 * 0.32))
    for ax, vals, label, col in zip(
        axes,
        [maes, rmses, mapes],
        ["MAE (passengers)", "RMSE (passengers)", "MAPE (%)"],
        [COLORS["blue"], COLORS["orange"], COLORS["green"]],
    ):
        ax.plot(horizons, vals, marker="o", color=col,
                markerfacecolor="white", markeredgewidth=1.0, linewidth=1.3, zorder=3)
        ax.fill_between(horizons, vals, min(vals), alpha=0.18, color=col)
        ax.axhline(min(vals), color=col, linewidth=0.7, linestyle=":", alpha=0.6)
        span   = max(vals) - min(vals)
        margin = max(span * 0.5, min(vals) * 0.02)
        ax.set_ylim(min(vals) - margin, max(vals) + margin)
        ax.set_xlabel("Forecast horizon (min)")
        ax.set_ylabel(label)
        ax.set_xticks(horizons)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout(pad=0.6)
    save(fig, "01_metrics_per_horizon.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 2 — Scatter predito x real
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(preds, targets, cfg):
    """
    [P4] Grid de subplots dinamico baseado em out_steps.
    Versao anterior: subplots(2,3) fixo para out_steps=2 -> 6 paineis,
    IndexError em axes.flatten()[h] para h>=2.
    Agora: ncols=min(out_steps,3), nrows=ceil(out_steps/ncols).
    Com out_steps=2: subplots(1,2) -- exatamente 2 paineis.

    [P1] Titulos com horizons via cfg.horizons_min — exibe [20, 40].
    """
    import matplotlib.colors as mcolors
    cmap = matplotlib.colormaps["viridis"]

    horizons = cfg.horizons_min   # [20, 40]

    all_true = [targets[:, h, :].ravel()[
        np.isfinite(targets[:, h, :].ravel()) & (targets[:, h, :].ravel() > 0)
    ] for h in range(cfg.out_steps)]
    all_true_cat = np.concatenate(all_true)
    lim_min = 0.0
    lim_max = float(np.percentile(all_true_cat, 99) * 1.05)

    rng     = np.random.default_rng(42)
    z_all, samples = [], []
    for h in range(cfg.out_steps):
        yt = targets[:, h, :].ravel(); yp = preds[:, h, :].ravel()
        valid = np.isfinite(yt) & np.isfinite(yp) & (yt > 0)
        yt, yp = yt[valid], yp[valid]
        idx = rng.choice(len(yt), size=min(8000, len(yt)), replace=False)
        yt_s, yp_s = yt[idx], yp[idx]
        try:
            z = gaussian_kde(np.vstack([yt_s, yp_s]), bw_method=0.15)(np.vstack([yt_s, yp_s]))
        except Exception:
            z = np.ones(len(yt_s))
        z_all.append(z); samples.append((yt_s, yp_s))

    z_min = min(z.min() for z in z_all)
    z_max = max(z.max() for z in z_all)
    norm  = mcolors.Normalize(vmin=z_min, vmax=z_max)

    # [P4] Grid dinamico: nunca mais paineis do que out_steps
    ncols = min(cfg.out_steps, 3)
    nrows = math.ceil(cfg.out_steps / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(COL2 * ncols / 3 + 0.6, COL2 * 0.35 * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for h in range(cfg.out_steps):
        ax      = axes_flat[h]
        yt_s, yp_s = samples[h]
        o = z_all[h].argsort()
        ax.scatter(yt_s[o], yp_s[o], c=z_all[h][o], s=2, cmap=cmap,
                   norm=norm, rasterized=True, linewidths=0)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], "r--", linewidth=0.8)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_aspect("equal")
        m  = compute_metrics(preds[:, h, :], targets[:, h, :])
        yt_f = targets[:, h, :].ravel(); yp_f = preds[:, h, :].ravel()
        v    = np.isfinite(yt_f) & np.isfinite(yp_f) & (yt_f > 0)
        r2   = float(np.corrcoef(yt_f[v], yp_f[v])[0, 1] ** 2)
        # [P1] titulo com horizonte real via cfg.horizons_min
        ax.set_title(f"t+{horizons[h]} min  |  MAE={m['MAE']:.2f}  R\u00b2={r2:.3f}")
        ax.set_xlabel("Observed (passengers)")
        ax.set_ylabel("Predicted (passengers)")

    # Ocultar paineis extras se nrows*ncols > out_steps
    for h in range(cfg.out_steps, nrows * ncols):
        axes_flat[h].set_visible(False)

    fig.subplots_adjust(right=0.88, hspace=0.45, wspace=0.35)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.025, 0.70])
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Point density (KDE)", fontsize=7, labelpad=6)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks([z_min, (z_min + z_max) / 2, z_max])
    cbar.set_ticklabels(["Low", "Medium", "High"])

    path = os.path.join(OUT_DIR, "02_scatter_pred_vs_real.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    log.info(f"Salvo: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figura 3 — Distribuicao do erro absoluto
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_distribution(preds, targets, cfg):
    """[P1] Labels dos horizontes via cfg.horizons_min — exibe [20, 40]."""
    horizons = [f"t+{hz} min" for hz in cfg.horizons_min]
    errors, p97 = [], 0.0
    for h in range(cfg.out_steps):
        err = np.abs(preds[:, h, :] - targets[:, h, :]).ravel()
        err = err[np.isfinite(err)]; errors.append(err)
        p97 = max(p97, float(np.percentile(err, 97)))

    fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.38))
    bp = ax.boxplot(errors, labels=horizons, patch_artist=True,
                    whis=(5, 95), showfliers=False,
                    medianprops=dict(color="black", linewidth=1.2),
                    whiskerprops=dict(linewidth=0.8),
                    capprops=dict(linewidth=0.8), widths=0.55)
    for patch, col in zip(bp["boxes"], C):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax.set_ylim(0, p97 * 1.08)
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Absolute error (passengers)")
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.text(0.99, 0.97, "Whiskers: 5th-95th percentile",
            transform=ax.transAxes, fontsize=6, color="gray", ha="right", va="top")
    fig.tight_layout(pad=0.6)
    save(fig, "03_error_distribution.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 4 — Serie temporal por parada
# ─────────────────────────────────────────────────────────────────────────────

def _build_artifact_mask(real: np.ndarray, freq_min: int) -> np.ndarray:
    """
    Detecta todos os trechos de imputacao ao longo de toda a serie temporal,
    nao apenas o prefixo inicial.

    PROBLEMA CORRIGIDO [P9]:
    A versao anterior varreva a serie ate encontrar a PRIMEIRA variacao
    (abs(v - ref_val) > 2.0) e marcava apenas o trecho inicial como
    artefato de boundary. Trechos constantes subsequentes — originados de:
      (a) periodos sem operacao (feriados, finais de semana, grade noturna)
          imputados com o valor da media do no (zero normalizado -> mu_n)
      (b) novas janelas de imputacao ffill/bfill dentro do split de teste
    — nao eram detectados, exibindo linha reta preta enganosa como se
    fossem observacoes reais.

    CORRECAO:
    Calcula a variancia local em janelas deslizantes de tamanho `window`
    (= in_steps = campo receptivo do modelo). Qualquer trecho onde a
    variancia local e inferior a VAR_THRESHOLD e marcado como artefato.
    O threshold VAR_THRESHOLD = 1e-3 e conservador: captura trechos onde
    o valor e praticamente constante (imputacao ou operacao absolutamente
    uniforme) sem marcar flutuacoes reais de baixa amplitude.

    Retorna: np.ndarray bool de shape (n_steps,) — True onde e artefato.
    """
    window        = max(4, 160 // freq_min)   # janela de in_steps (8 para freq_min=20)
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
    Sombreia em cinza todos os intervalos contiguos onde artifact=True,
    traca uma linha vertical tracejada em cada fronteira e insere o label
    "Imputed data" no primeiro intervalo encontrado (se label_first=True).
    """
    # Encontra intervalos contiguos de artifact=True
    padded    = np.concatenate([[False], artifact, [False]])
    starts    = np.where(~padded[:-1] &  padded[1:])[0]
    ends      = np.where( padded[:-1] & ~padded[1:])[0]

    first_labeled = False
    for s, e in zip(starts, ends):
        t_s = time_axis[s]
        t_e = time_axis[min(e, len(time_axis) - 1)]
        ax.axvspan(t_s, t_e, color="gray", alpha=0.13, zorder=5, linewidth=0)
        ax.axvline(t_s, color="gray", linewidth=0.5, linestyle="--", alpha=0.6, zorder=6)
        ax.axvline(t_e, color="gray", linewidth=0.5, linestyle="--", alpha=0.6, zorder=6)
        if label_first and not first_labeled:
            mid   = (t_s + t_e) / 2
            y_pos = real[s:e].mean() if (e > s) else real[s]
            ax.text(mid, y_pos,
                    "Imputed data",
                    fontsize=5.5, color="gray", ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            first_labeled = True


def _plot_one_stop(preds, targets, cfg, local_idx, fname,
                   global_indices, node_meta, n_steps=96):
    """
    [P1] horizons via cfg.horizons_min — exibe [20, 40] para freq_min=20.
    [P2] time_axis = np.arange(n_steps) * cfg.freq_min.
    [P9] Deteccao de artefatos de imputacao em TODA a serie (correcao):
         A versao anterior so detectava o trecho inicial constante (primeiro
         boundary). Trechos planos subsequentes — periodos sem operacao
         imputados com zero normalizado (que desnormaliza para mu_n ~ 135 pass.)
         — apareciam como linha preta reta enganosa em regioes brancas do grafico,
         sugerindo observacoes reais de valor constante quando na verdade eram
         artefatos de imputacao.
         Correcao: _build_artifact_mask varre TODA a serie com janela deslizante
         e detecta qualquer trecho de baixa variancia local (var < 1e-3), seja no
         inicio, meio ou fim. _shade_artifact_regions sombreia todos esses trechos.
         MAE reportado no titulo do painel calculado apenas sobre timesteps reais
         (artifact=False), evitando inflacao da metrica por trechos imputados.
    """
    horizons  = cfg.horizons_min
    time_axis = np.arange(n_steps) * cfg.freq_min
    hcolors   = [COLORS["blue"], COLORS["sky"], COLORS["green"],
                 COLORS["orange"], COLORS["purple"], COLORS["red"]]

    if cfg.out_steps == 1:
        fig, axes_list = plt.subplots(1, 1, figsize=(COL2, COL2 * 0.28))
        axes_list = [axes_list]
    else:
        fig, axes_arr = plt.subplots(cfg.out_steps, 1,
                                     figsize=(COL2, COL2 * 0.22 * cfg.out_steps),
                                     sharex=True)
        axes_list = list(axes_arr)

    real_full = targets[:n_steps, 0, local_idx]

    # [P9] Detecta artefatos em toda a serie — nao apenas no prefixo.
    artifact  = _build_artifact_mask(real_full, cfg.freq_min)
    real_mask = ~artifact   # True onde os dados sao reais

    n_real    = int(real_mask.sum())
    n_total   = len(real_full)
    pct_real  = n_real / max(n_total, 1) * 100
    log.info(
        f"  Parada local_idx={local_idx}: "
        f"{n_real}/{n_total} timesteps reais ({pct_real:.1f}%), "
        f"{n_total - n_real} imputados"
    )

    for h, (ax, hz, col) in enumerate(zip(axes_list, horizons, hcolors)):
        real = targets[:n_steps, 0, local_idx]
        pred = preds[:n_steps, h, local_idx]

        ax.plot(time_axis, real, color=COLORS["black"], linewidth=0.85,
                label="Observed", zorder=4)
        ax.plot(time_axis, pred, color=col, linewidth=0.9, linestyle="--",
                label=f"Predicted (t+{hz} min)", zorder=3)
        ax.fill_between(time_axis, real, pred, alpha=0.15, color=col, zorder=2)

        # [P9] MAE calculado apenas sobre timesteps reais (artifact=False).
        # Se nenhum timestep real existir, reporta nan.
        if real_mask.any():
            mae_local = float(np.nanmean(np.abs(pred[real_mask] - real[real_mask])))
            mae_label = f"{mae_local:.2f} pass."
        else:
            mae_label = "n/a (sem dados reais)"

        ax.text(0.01, 0.97,
                f"t+{hz} min  |  MAE = {mae_label}  "
                f"[{pct_real:.0f}% dados reais]",
                transform=ax.transAxes, fontsize=7, va="top")

        # [P9] Sombreia TODOS os trechos de artefato, nao apenas o inicial.
        _shade_artifact_regions(ax, time_axis, artifact, real,
                                 label_first=(h == 0))

        ax.set_ylabel("Loading\n(pass.)")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        if h == 0:
            ax.legend(loc="upper right", ncol=2, handlelength=1.4, fontsize=7)

    axes_list[-1].set_xlabel("Time (min)")
    axes_list[-1].xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.suptitle(
        f"{_node_label(local_idx, global_indices, node_meta)} -- all forecast horizons",
        fontsize=8, y=1.005
    )
    fig.tight_layout(pad=0.4)
    save(fig, fname)


def plot_temporal_forecast(preds, targets, cfg, active_mask,
                           global_indices, node_meta, n_stops=3, n_steps=96):
    with np.errstate(all="ignore"):
        mean_demand = np.nanmean(targets, axis=(0, 1))
    valid_local = np.where(np.isfinite(mean_demand))[0]
    top_local   = valid_local[np.argsort(mean_demand[valid_local])[-n_stops:][::-1]]
    fnames = ["04a_temporal_stop_1.pdf", "04b_temporal_stop_2.pdf", "04c_temporal_stop_3.pdf"]
    for local_idx, fname in zip(top_local, fnames):
        _plot_one_stop(preds, targets, cfg, local_idx, fname,
                       global_indices, node_meta, n_steps)


# ─────────────────────────────────────────────────────────────────────────────
# Figura 5 — MAE por parada
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_error(preds, targets, cfg, top_n=20):
    with np.errstate(all="ignore"):
        mae_per_stop = np.nanmean(np.abs(preds - targets), axis=(0, 1))
    valid = np.where(np.isfinite(mae_per_stop))[0]
    rank  = np.argsort(mae_per_stop[valid])
    top_n = min(top_n, len(valid))

    fig, axes = plt.subplots(1, 2, figsize=(COL2, COL2 * 0.38))
    for ax, local_idx, title, col in zip(
        axes,
        [rank[-top_n:][::-1], rank[:top_n]],
        [f"Top-{top_n} highest MAE stops", f"Top-{top_n} lowest MAE stops"],
        [COLORS["red"], COLORS["green"]],
    ):
        global_idx = valid[local_idx]
        ypos       = np.arange(top_n)
        ax.barh(ypos, mae_per_stop[global_idx], color=col, alpha=0.75, height=0.7)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"S{i}" for i in global_idx], fontsize=6)
        ax.set_xlabel("Mean absolute error (passengers)")
        ax.set_title(title)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.invert_yaxis()
    fig.tight_layout(pad=0.6)
    save(fig, "05_spatial_error_map.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Figura 6 — CDF
# ─────────────────────────────────────────────────────────────────────────────

def plot_cdf(preds, targets, cfg):
    """
    [P5] Labels das curvas via cfg.horizons_min — exibe "t+20 min" e "t+40 min".
    x_max rastreado durante o plot; set_xlim aplicado antes da anotacao
    para evitar posicionamento erroneo do texto "90th pct.".
    """
    horizons = cfg.horizons_min   # [20, 40]
    fig, ax  = plt.subplots(figsize=(COL1, COL1 * 0.85))
    x_max = 0.0
    for h in range(cfg.out_steps):
        err = np.abs(preds[:, h, :] - targets[:, h, :]).ravel()
        err = err[np.isfinite(err)]
        sorted_err = np.sort(err)
        cdf        = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        step       = max(1, len(sorted_err) // 2000)
        # [P5] label com horizonte real via cfg.horizons_min
        ax.plot(sorted_err[::step], cdf[::step], color=C[h],
                linewidth=1.0, label=f"t+{horizons[h]} min")
        x_max = max(x_max, float(sorted_err[-1]))

    ax.set_xlim(left=0, right=x_max * 1.02)
    ax.set_ylim(0, 1)
    ax.axhline(0.90, color="gray", linewidth=0.7, linestyle=":", alpha=0.8)
    ax.text(x_max * 0.03, 0.915, "90th pct.", fontsize=7, color="gray")
    ax.set_xlabel("Absolute error (passengers)")
    ax.set_ylabel("Cumulative probability")
    ax.legend(title="Horizon", loc="lower right")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout(pad=0.5)
    save(fig, "06_cumulative_error.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=== GraphWaveNet -- Geracao de Figuras (Qualis A1) ===")
    preds, targets, cfg, active_mask, node_meta, global_indices = load_predictions()
    log.info(f"horizons: {cfg.horizons_min} | out_steps: {cfg.out_steps}")
    log.info("Gerando figuras (apenas nos ativos)...")
    plot_metrics_per_horizon(preds, targets, cfg)
    plot_scatter(preds, targets, cfg)
    plot_error_distribution(preds, targets, cfg)
    plot_temporal_forecast(preds, targets, cfg, active_mask, global_indices, node_meta)
    plot_spatial_error(preds, targets, cfg)
    plot_cdf(preds, targets, cfg)
    log.info(f"Todas as figuras salvas em: {OUT_DIR}")