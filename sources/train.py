# train.py — versao corrigida para pipeline v7 (janela 20 min)
#
# Alteracoes em relacao a versao anterior:
#
# [T1] cfg.lr_patience em vez de cfg.patience // 2:
#      A propriedade lr_patience foi adicionada ao Config em [A6].
#
# [T2] Log de horizontes via cfg.horizons_min:
#      Com freq_min=20: exibe "t+20min | t+40min".
#
# [T3] Log de in_steps e out_steps com contexto temporal.
#
# [T4] horizon_weights ativado na loss (CORRECAO):
#      cfg.horizon_weights e convertido para tensor e passado para
#      weighted_masked_mae_multihorizon() em train_epoch e eval_epoch.
#      Anteriormente a funcao aceitava o parametro mas nunca o recebia,
#      tornando a ponderacao por horizonte inoperante em ambas as fases.
#      O tensor e criado uma vez fora dos loops e reutilizado.
#
# [T5] build_demand_weights corrigido (IC-3):
#      Os pesos de demanda agora sao calculados a partir de train_mean.csv
#      (escala original, passageiros/janela), e nao sobre X_train normalizado.
#      Sobre o tensor z-score, mean(loading) ~ 0 para todos os nos, tornando
#      (mean + eps)^beta ~ eps^beta — pesos praticamente uniformes e o efeito
#      de ponderacao por demanda era nulo. A correcao usa train_mean["loading"]
#      reindexado por node_order para garantir alinhamento com o tensor.
#
# [T6] make_loaders chamado sem construir test_loader desnecessario (IC-4):
#      train.py nao usa o test_loader. A chamada anterior passava X_test inteiro
#      e construia um DataLoader que nunca era iterado. Agora X_test=None e
#      make_loaders retorna None no terceiro elemento quando X_test e None.
#      Requer suporte a None em dataset.make_loaders (veja docstring abaixo).
#
# [T7] Checkpoint salva node_order para validacao de alinhamento em evaluate.py:
#      O campo "node_order" e adicionado ao dict salvo, permitindo que
#      evaluate.py verifique consistencia entre o modelo treinado e os tensores
#      de teste sem depender de convencao implicita de ordem de nos.

import os
import logging
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
        logging.FileHandler("training.log", encoding="utf-8"),
        _stream_handler,
    ]
)
log = logging.getLogger(__name__)


def build_demand_weights(train_mean_path: str, node_order: list, cfg) -> torch.FloatTensor:
    """
    Calcula pesos de demanda w_n = (mean_n + eps)^beta, normalizados para
    media = 1. Retorna tensor de shape (N,).

    [T5] CORRECAO IC-3: os pesos sao calculados sobre train_mean.csv
    (escala original de passageiros), NAO sobre X_train normalizado.
    Sobre o tensor z-score, mean(loading) ~ 0 para todos os nos, fazendo
    (mean + eps)^beta ~ constante e anulando a diferenciacaoo entre nos.

    Args:
        train_mean_path : caminho para train_mean.csv gerado pelo pipeline.
        node_order      : lista de node_ids na mesma ordem do eixo N do tensor.
        cfg             : objeto Config com target_col_idx e demand_weight_beta.
    """
    train_mean = pd.read_csv(train_mean_path, index_col=0)

    # Reindexar por node_order garante alinhamento com o tensor (N, F).
    # Nos ausentes em train_mean recebem mean=0 (comportamento conservador).
    means = train_mean["loading"].reindex(node_order, fill_value=0.0).values.astype(np.float32)

    n_missing = int((train_mean["loading"].reindex(node_order).isna()).sum())
    if n_missing > 0:
        log.warning(
            f"build_demand_weights: {n_missing} nos ausentes em train_mean.csv "
            f"— media imputada como 0.0 (peso minimo)."
        )

    weights = (means + 1e-6) ** cfg.demand_weight_beta
    weights = weights / weights.mean()

    log.info(
        f"Pesos demanda    — fonte: train_mean.csv (escala original) | "
        f"beta={cfg.demand_weight_beta} | "
        f"min={weights.min():.3f} | max={weights.max():.3f} | "
        f"razao={weights.max()/weights.min():.1f}x"
    )
    return torch.FloatTensor(weights)


def weighted_masked_mae_multihorizon(pred, target, dw, horizon_weights=None):
    """
    MAE ponderado por demanda e por horizonte, com mascara de zeros.

    Args:
        pred             : (B, T_out, N)
        target           : (B, T_out, N)
        dw               : (N,) — pesos de demanda por no
        horizon_weights  : (T_out,) — pesos por horizonte de predicao.
                           Ex: [2.0, 1.0] prioriza t+20 sobre t+40.
                           None equivale a pesos unitarios.
    """
    if horizon_weights is None:
        horizon_weights = torch.ones(pred.shape[1], device=pred.device)
    hw    = horizon_weights[None, :, None]   # (1, T_out, 1)
    mask  = (target != 0.0).float()
    w     = dw[None, None, :]                # (1, 1,     N)
    loss  = torch.abs(pred - target) * mask * w * hw
    denom = (mask * w * hw).sum().clamp(min=1)
    return loss.sum() / denom


def train_epoch(model, loader, optimizer, scaler, device, cfg, dw, hw):
    """
    hw : tensor (T_out,) de horizon_weights ja alocado no device correto.
    """
    model.train()
    total_loss  = 0.0
    device_type = device.type

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
            # [T4] hw passado explicitamente — ponderacao por horizonte ativa.
            loss = weighted_masked_mae_multihorizon(model(X), y, dw, horizon_weights=hw)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, device, cfg, dw, hw):
    """
    hw : tensor (T_out,) de horizon_weights ja alocado no device correto.
    """
    model.eval()
    total_loss  = 0.0
    device_type = device.type

    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device_type, enabled=cfg.use_amp):
            # [T4] hw passado explicitamente — ponderacao por horizonte ativa.
            total_loss += weighted_masked_mae_multihorizon(
                model(X), y, dw, horizon_weights=hw
            ).item()

    return total_loss / len(loader)


def run_training():
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and cfg.device == "cuda":
        log.warning("CUDA solicitado mas nao disponivel — usando CPU.")

    horizons_str = ", ".join(f"t+{h}min" for h in cfg.horizons_min)
    hw_str       = ", ".join(str(w) for w in cfg.horizon_weights)
    log.info(f"Dispositivo      : {device}")
    log.info(f"AMP              : {cfg.use_amp}")
    log.info(f"batch_size       : {cfg.batch_size}")
    log.info(f"num_layers       : {cfg.num_layers} | num_adj: {cfg.num_adj}")
    log.info(
        f"Janela           : {cfg.freq_min} min | "
        f"in_steps={cfg.in_steps} ({cfg.in_steps*cfg.freq_min}min) | "
        f"out_steps={cfg.out_steps} ({horizons_str})"
    )
    # [T4] Loga os pesos de horizonte ativos para rastreabilidade no log.
    log.info(
        f"horizon_weights  : [{hw_str}] "
        f"— {horizons_str.replace(', ', ' / ')}"
    )

    X_train  = np.load(os.path.join(cfg.ARTIFACTS_DIR, "X_train.npy"))
    X_val    = np.load(os.path.join(cfg.ARTIFACTS_DIR, "X_val.npy"))
    adj_geo  = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_geo.npy"))
    adj_topo = np.load(os.path.join(cfg.ARTIFACTS_DIR, "adj_topo.npy"))

    cfg.num_nodes = X_train.shape[1]
    N = cfg.num_nodes

    log.info(f"Paradas (N)      : {N} | Features (F): {X_train.shape[2]}")
    log.info(
        f"Timesteps        — treino: {X_train.shape[0]} "
        f"({X_train.shape[0]*cfg.freq_min/60:.1f}h) | "
        f"val: {X_val.shape[0]}"
    )

    # Verifica consistencia entre tensor e config — falha explicita.
    assert X_train.shape[2] == cfg.in_features, (
        f"in_features no tensor ({X_train.shape[2]}) != cfg ({cfg.in_features}). "
        f"Verifique ALL_FEATURES no pipeline e in_features em config.py."
    )

    # Recupera node_order a partir de node_metadata.csv para alinhamento com
    # train_mean.csv e para salvar no checkpoint (usado em evaluate.py).
    node_meta_path  = os.path.join(cfg.ARTIFACTS_DIR, "node_metadata.csv")
    node_meta       = pd.read_csv(node_meta_path).sort_values("node_index")
    node_order      = node_meta["node_id"].tolist()
    assert len(node_order) == N, (
        f"node_metadata.csv tem {len(node_order)} nos, tensor tem N={N}. "
        f"Regenere os artefatos com o pipeline."
    )

    # [T5] build_demand_weights agora usa train_mean.csv (escala real).
    train_mean_path = os.path.join(cfg.ARTIFACTS_DIR, "train_mean.csv")
    dw = build_demand_weights(train_mean_path, node_order, cfg).to(device)

    # [T4] Converte horizon_weights para tensor no device correto.
    #      Criado uma vez aqui e passado para train_epoch e eval_epoch,
    #      evitando realocacao a cada batch.
    hw = torch.tensor(cfg.horizon_weights, dtype=torch.float32, device=device)

    # [T6] X_test=None evita construir DataLoader nao utilizado em train.py.
    #      make_loaders deve retornar None no terceiro elemento quando X_test=None.
    train_loader, val_loader, _ = make_loaders(X_train, X_val, None, cfg)

    X_b, y_b = next(iter(train_loader))
    assert X_b.shape == (cfg.batch_size, cfg.in_steps, N, cfg.in_features), \
        f"Shape X inesperado: {tuple(X_b.shape)}"
    assert y_b.shape == (cfg.batch_size, cfg.out_steps, N), \
        f"Shape y inesperado: {tuple(y_b.shape)}"
    log.info(f"Shapes OK        — X: {tuple(X_b.shape)} | y: {tuple(y_b.shape)}")

    model     = GraphWaveNet(cfg, adj_geo, adj_topo).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # [T1] cfg.lr_patience em vez de cfg.patience // 2 hardcoded.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=cfg.lr_patience,
        min_lr=1e-6
    )
    scaler = torch.amp.GradScaler(
        device  = device.type,
        enabled = (cfg.use_amp and device.type == "cuda")
    )

    best_val, no_improve = float("inf"), 0

    log.info(f"\n{'Epoca':>6} {'Train MAE':>12} {'Val MAE':>10} {'LR':>12}")
    log.info("-" * 46)

    for epoch in range(1, cfg.epochs + 1):
        # [T4] hw passado para ambas as fases de treino e validacao.
        train_loss = train_epoch(model, train_loader, optimizer,
                                 scaler, device, cfg, dw, hw)
        val_loss   = eval_epoch(model, val_loader, device, cfg, dw, hw)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        log.info(f"{epoch:>6}   {train_loss:>10.4f}   {val_loss:>8.4f}   {lr_now:>12.2e}")

        if val_loss < best_val:
            best_val, no_improve = val_loss, 0
            torch.save({
                "epoch"           : epoch,
                "model_state"     : model.state_dict(),
                "optimizer_state" : optimizer.state_dict(),
                "scheduler_state" : scheduler.state_dict(),
                "scaler_state"    : scaler.state_dict(),
                "best_val"        : best_val,
                "freq_min"        : cfg.freq_min,
                "in_steps"        : cfg.in_steps,
                "out_steps"       : cfg.out_steps,
                "in_features"     : cfg.in_features,
                "horizon_weights" : cfg.horizon_weights,
                # [T7] node_order salvo para validacao de alinhamento em evaluate.py.
                "node_order"      : node_order,
            }, cfg.checkpoint_path)
            log.info(f"         -> checkpoint salvo (val MAE: {best_val:.4f})")
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                log.info(
                    f"\nEarly stopping na epoca {epoch}. "
                    f"Melhor val MAE: {best_val:.4f}"
                )
                break

    log.info(f"\nTreinamento concluido. Melhor val MAE: {best_val:.4f}")
    return model, cfg, adj_geo, adj_topo


if __name__ == "__main__":
    run_training()