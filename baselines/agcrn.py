"""
agcrn.py
========
AGCRN -- Adaptive Graph Convolutional Recurrent Network
Bai et al., NeurIPS 2020. https://arxiv.org/abs/2007.02842

Correcoes de assimetria em relacao ao GWN (v3):

[A1] AGCRNWrapper: lr_patience via cfg.lr_patience em vez de
     max(1, cfg.patience // 2) hardcoded.

[A2] Checkpoint do AGCRN: salva freq_min, in_steps, out_steps,
     in_features e node_order — simetrico com o checkpoint do GWN.
     [AG-4 CORRIGIDO]: versao anterior nao salvava node_order nem
     in_features, impedindo validacao de alinhamento em recarregamentos.

[A3] Log de contexto temporal no inicio do treinamento.

[A4] hidden_dim e num_layers lidos de cfg.

[A5] horizon_weights na loss (AG-1 — CRITICO):
     _weighted_masked_mae_multihorizon substituiu _weighted_masked_mae.
     A versao anterior ignorava cfg.horizon_weights ([2.0, 1.0]),
     treinando o AGCRN sem pressao sobre t+20 min enquanto o GWN
     aplicava peso 2x nesse horizonte. A comparacao MAE por horizonte
     era invalida: modelos treinados com funcoes de perda assimetricas.
     Correcao: mesma logica de ponderacao de horizon_weights usada em
     train.py — hw tensor criado uma vez e passado para train/eval.

[A6] Bias separadas por gate — AGCRNCell (AG-5 — MODERADO):
     A versao anterior usava um unico self.bias adicionado em ambas as
     chamadas _dagg_gcn_chunked (para xh e para xrh). O AGCRN original
     (Bai et al. 2020) usa parametros de bias distintos para cada gate.
     Correcao: self.bias_xh e self.bias_xrh, um por chamada.

[A7] make_loaders no _agcrn_worker: X_test passado como None apos
     inferencia concluida com test_loader (AG-2 / IC-D1):
     O worker chamava make_loaders(X_train, X_val, X_test, cfg) alocando
     os tres arrays completos simultaneamente em RAM antes de liberar
     qualquer um. Com N=18.340 e tres splits, pico desnecessario de ~4-6 GB.
     Correcao: make_loaders chamado duas vezes — primeiro para treino/val
     (X_test=None) e depois separadamente para o test_loader (X_train=None,
     X_val=None), permitindo que X_train e X_val sejam coletados pelo GC
     antes de carregar o test.
     Nota: o _agcrn_worker nao controla quando X_train/X_val sao recebidos
     (passados como argumentos do processo), mas a mudanca no make_loaders
     evita que os DataLoaders retidos dupliquem o consumo de RAM durante o
     treinamento.
"""

import os
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

log = logging.getLogger(__name__)

# Defaults de modulo — usados apenas quando AGCRN e instanciado fora
# do pipeline (standalone). Dentro do pipeline, cfg sobrescreve estes valores.
AGCRN_HIDDEN_DIM = 32
AGCRN_EMB_DIM    = 10
AGCRN_NUM_LAYERS = 2
AGCRN_DROPOUT    = 0.2
AGCRN_CHUNK      = 1024   # alinhado com CHUNK=256 do GWN (_adapt_diff_ste)


# ─────────────────────────────────────────────────────────────────────────────
# Funcao auxiliar de chunk — encapsulada para gradient checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def _gcn_chunk_with_slice(E_chunk: torch.Tensor, E_full: torch.Tensor,
                           x_full: torch.Tensor, x_slice: torch.Tensor,
                           W_pool: torch.Tensor,
                           C_in: int, H3: int) -> torch.Tensor:
    """
    Processa UM chunk de nos com DAGG e gradient checkpointing.

    E_chunk : (chunk, D)
    E_full  : (N, D)
    x_full  : (B, N, C_in)
    x_slice : (B, chunk, C_in)
    W_pool  : (D, C_in*H3)
    Retorna : (B, chunk, H3)
    """
    chunk_size = E_chunk.shape[0]
    W_chunk    = (E_chunk @ W_pool).view(chunk_size, C_in, H3)

    with torch.no_grad():
        scores  = F.relu(E_chunk @ E_full.T)
        A_chunk = F.softmax(scores, dim=-1).detach()

    x_agg     = torch.einsum("rn,bnc->brc", A_chunk, x_full)
    agg_chunk = torch.einsum("brc,rch->brh", x_agg, W_chunk)
    res_chunk = torch.einsum("brc,rch->brh", x_slice, W_chunk)

    return agg_chunk + res_chunk


# ─────────────────────────────────────────────────────────────────────────────
# AGCRNCell
# ─────────────────────────────────────────────────────────────────────────────

class AGCRNCell(nn.Module):
    """
    Celula GRU com convolucao em grafo adaptativa (DAGG).

    [A6] CORRECAO AG-5: dois parametros de bias separados — bias_xh e
    bias_xrh — em vez de um unico self.bias compartilhado entre os dois
    gates. O AGCRN original (Bai et al. 2020, Eq. 5-7) usa parametros
    de bias distintos para o gate de entrada/reset e para o gate de
    candidato. Compartilhar um unico bias reduz a expressividade do
    modelo: o gradiente da primeira chamada (xh) e o da segunda (xrh)
    acumulam-se no mesmo tensor, acoplando os dois gates de forma nao
    intencional.
    """

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int,
                 num_nodes: int, chunk: int = AGCRN_CHUNK):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes  = num_nodes
        self.chunk      = chunk
        self.C_in       = in_dim + hidden_dim
        self.H3         = hidden_dim * 3

        self.W_pool   = nn.Parameter(torch.empty(emb_dim, self.C_in * self.H3))
        # [A6] Dois bias separados — um por chamada a _dagg_gcn_chunked.
        self.bias_xh  = nn.Parameter(torch.zeros(self.H3))  # gate Z/R (entrada + h)
        self.bias_xrh = nn.Parameter(torch.zeros(self.H3))  # gate C  (entrada + R*h)
        nn.init.xavier_uniform_(self.W_pool)

    def _dagg_gcn_chunked(self, E: torch.Tensor, x: torch.Tensor,
                           b: torch.Tensor) -> torch.Tensor:
        """
        GCN com DAGG em chunks de `self.chunk` nos.
        Gradient checkpointing por chunk limita o pico de memoria.
        """
        B, N, _ = x.shape
        out = torch.zeros(B, N, self.H3, device=x.device, dtype=x.dtype)

        for i0 in range(0, N, self.chunk):
            i1      = min(i0 + self.chunk, N)
            E_chunk = E[i0:i1]
            x_slice = x[:, i0:i1, :].contiguous()

            chunk_out = grad_ckpt(
                _gcn_chunk_with_slice,
                E_chunk, E, x, x_slice, self.W_pool,
                self.C_in, self.H3,
                use_reentrant=False,
            )
            out[:, i0:i1, :] = chunk_out

        return out + b

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                E: torch.Tensor) -> torch.Tensor:
        xh  = torch.cat([x, h], dim=-1)
        # [A6] bias_xh para o gate Z/R; bias_xrh para o gate C.
        out = self._dagg_gcn_chunked(E, xh, self.bias_xh)

        H            = self.hidden_dim
        Z_pre, R_pre, _ = out.split(H, dim=-1)
        Z = torch.sigmoid(Z_pre)
        R = torch.sigmoid(R_pre)

        xrh   = torch.cat([x, R * h], dim=-1)
        out_c = self._dagg_gcn_chunked(E, xrh, self.bias_xrh)
        _, _, C_pre = out_c.split(H, dim=-1)
        C = torch.tanh(C_pre)

        return (1.0 - Z) * h + Z * C


# ─────────────────────────────────────────────────────────────────────────────
# AGCRNLayer
# ─────────────────────────────────────────────────────────────────────────────

class AGCRNLayer(nn.Module):
    """
    Layer recorrente que itera sobre in_steps aplicando AGCRNCell.
    dropout=0.0 na ultima layer.
    """

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int,
                 num_nodes: int, dropout: float = 0.2,
                 chunk: int = AGCRN_CHUNK):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell       = AGCRNCell(in_dim, hidden_dim, emb_dim, num_nodes, chunk)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, E: torch.Tensor,
                h0: Optional[torch.Tensor] = None):
        B, T, N, _ = x.shape
        if h0 is None:
            h0 = torch.zeros(B, N, self.hidden_dim, device=x.device, dtype=x.dtype)

        h, outputs = h0, []
        for t in range(T):
            h = self.cell(x[:, t], h, E)
            outputs.append(h)

        return self.dropout(torch.stack(outputs, dim=1)), h


# ─────────────────────────────────────────────────────────────────────────────
# AGCRN
# ─────────────────────────────────────────────────────────────────────────────

class AGCRN(nn.Module):
    """
    AGCRN completo.

    Entrada : (B, T_in,  N, F)  — identico ao GWN
    Saida   : (B, T_out, N)     — identico ao GWN

    [A4] hidden_dim e num_layers lidos de cfg.
    """

    def __init__(self, cfg,
                 hidden_dim: Optional[int]   = None,
                 emb_dim:    int              = AGCRN_EMB_DIM,
                 num_layers: Optional[int]   = None,
                 dropout:    float            = AGCRN_DROPOUT,
                 chunk:      int              = AGCRN_CHUNK):
        super().__init__()

        hidden_dim  = cfg.hidden_dim  if hidden_dim  is None else hidden_dim
        num_layers  = cfg.num_layers  if num_layers  is None else num_layers

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        N               = cfg.num_nodes

        self.node_emb   = nn.Embedding(N, emb_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        self.input_proj = nn.Linear(cfg.in_features, hidden_dim)

        self.layers = nn.ModuleList([
            AGCRNLayer(
                in_dim     = hidden_dim,
                hidden_dim = hidden_dim,
                emb_dim    = emb_dim,
                num_nodes  = N,
                dropout    = dropout if i < num_layers - 1 else 0.0,
                chunk      = chunk,
            )
            for i in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cfg.out_steps),
        )

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info(
            f"[AGCRN] N={N} | F={cfg.in_features} | hidden={hidden_dim} | "
            f"emb={emb_dim} | layers={num_layers} | chunk={chunk} | "
            f"params={n_params:,}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, T_in, N, F) -> (B, T_out, N)"""
        x = self.input_proj(x)
        E = self.node_emb.weight

        h: Optional[torch.Tensor] = None
        for layer in self.layers:
            x, h = layer(x, E, h)

        return self.output_proj(h).permute(0, 2, 1)


# ─────────────────────────────────────────────────────────────────────────────
# AGCRNWrapper
# ─────────────────────────────────────────────────────────────────────────────

class AGCRNWrapper:
    """
    Encapsula treinamento, validacao e inferencia do AGCRN.
    Interface simetrica com o loop de treinamento do GWN (train.py).
    """

    def __init__(self, cfg, device: torch.device, use_amp: bool,
                 demand_weights: torch.Tensor,
                 node_order: Optional[list] = None):
        self.cfg            = cfg
        self.device         = device
        self.use_amp        = use_amp and device.type == "cuda"
        self.device_type    = device.type
        self.demand_weights = demand_weights.to(device)
        self.node_order     = node_order   # [A2] salvo no checkpoint
        self.model          = AGCRN(cfg).to(device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr           = cfg.lr,
            weight_decay = cfg.weight_decay,
        )
        # [A1] cfg.lr_patience — simetrico com GWN.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5,
            patience=cfg.lr_patience,
            min_lr=1e-6,
        )
        self.scaler = torch.amp.GradScaler(
            device=self.device_type, enabled=self.use_amp
        )
        # [A5] hw tensor criado uma vez — simetrico com train.py [T4].
        self.hw = torch.tensor(
            cfg.horizon_weights, dtype=torch.float32, device=device
        )

    @staticmethod
    def _weighted_masked_mae_multihorizon(pred, target, dw,
                                           horizon_weights=None,
                                           null_val=0.0):
        """
        [A5] CORRECAO AG-1: MAE ponderado por demanda E por horizonte,
        identico a weighted_masked_mae_multihorizon de train.py.

        pred            : (B, T_out, N)
        target          : (B, T_out, N)
        dw              : (N,) — pesos de demanda
        horizon_weights : (T_out,) — pesos por horizonte
                          Ex: [2.0, 1.0] prioriza t+20 sobre t+40.
                          None = pesos unitarios.
        """
        if horizon_weights is None:
            horizon_weights = torch.ones(pred.shape[1], device=pred.device)
        hw    = horizon_weights[None, :, None]   # (1, T_out, 1)
        mask  = (target != null_val).float()
        w     = dw[None, None, :]                # (1, 1,     N)
        loss  = torch.abs(pred - target) * mask * w * hw
        denom = (mask * w * hw).sum().clamp(min=1)
        return loss.sum() / denom

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total = 0.0
        for X, y in loader:
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
                # [A5] hw passado explicitamente — ponderacao por horizonte ativa.
                loss = self._weighted_masked_mae_multihorizon(
                    self.model(X), y, self.demand_weights,
                    horizon_weights=self.hw,
                )
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item()
            del X, y, loss
        return total / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader) -> float:
        self.model.eval()
        total = 0.0
        for X, y in loader:
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
                # [A5] hw passado explicitamente.
                total += self._weighted_masked_mae_multihorizon(
                    self.model(X), y, self.demand_weights,
                    horizon_weights=self.hw,
                ).item()
            del X, y
        return total / len(loader)

    def train(self, train_loader, val_loader, checkpoint_dir: str) -> "AGCRNWrapper":
        cfg  = self.cfg
        ckpt = os.path.join(checkpoint_dir, "agcrn_best.pt")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # [A3] Log de contexto temporal — simetrico com train.py (GWN)
        horizons_str = ", ".join(f"t+{h}min" for h in cfg.horizons_min)
        hw_str       = ", ".join(str(w) for w in cfg.horizon_weights)
        log.info(
            f"\n[AGCRN] Treinando -- ate {cfg.epochs} epocas | "
            f"device: {self.device_type} | AMP: {self.use_amp}\n"
            f"  Janela: {cfg.freq_min}min | "
            f"in_steps={cfg.in_steps} ({cfg.in_steps*cfg.freq_min}min) | "
            f"out_steps={cfg.out_steps} ({horizons_str})\n"
            f"  hidden={self.model.hidden_dim} | layers={self.model.num_layers} | "
            f"chunk={AGCRN_CHUNK} | beta={cfg.demand_weight_beta} | "
            f"lr_patience={cfg.lr_patience}\n"
            f"  horizon_weights: [{hw_str}] — {horizons_str.replace(', ', ' / ')}"
        )
        log.info(f"  {'Epoca':>6} {'Train MAE':>12} {'Val MAE':>10} {'LR':>12}")
        log.info(f"  {'-'*6} {'-'*12} {'-'*10} {'-'*12}")

        best_val, no_improve = float("inf"), 0

        for epoch in range(1, cfg.epochs + 1):
            t_loss = self._train_epoch(train_loader)
            v_loss = self._eval_epoch(val_loader)
            self.scheduler.step(v_loss)
            lr_now = self.optimizer.param_groups[0]["lr"]

            # Log de todas as epocas — simetrico com train.py (GWN).
            # A versao anterior filtrava com "epoch % 5 == 0 or epoch == 1",
            # causando saltos no log e checkpoint sem numero de epoca visivel.
            log.info(
                f"  {epoch:>6}   {t_loss:>10.4f}   {v_loss:>8.4f}   {lr_now:>12.2e}"
            )

            if v_loss < best_val:
                best_val, no_improve = v_loss, 0
                # [A2] Salva node_order e in_features — simetrico com GWN [T7].
                torch.save({
                    "epoch"          : epoch,
                    "model_state"    : self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "scaler_state"   : self.scaler.state_dict(),
                    "best_val"       : best_val,
                    "freq_min"       : cfg.freq_min,
                    "in_steps"       : cfg.in_steps,
                    "out_steps"      : cfg.out_steps,
                    "in_features"    : cfg.in_features,
                    "horizon_weights": cfg.horizon_weights,
                    "node_order"     : self.node_order,
                }, ckpt)
                log.info(f"         -> checkpoint salvo (val MAE: {best_val:.4f})")
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    log.info(
                        f"\n[AGCRN] Early stopping na epoca {epoch}. "
                        f"Melhor val MAE: {best_val:.4f}"
                    )
                    break

        log.info(f"[AGCRN] Concluido. Melhor val MAE: {best_val:.4f}")
        ckpt_data = torch.load(ckpt, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt_data["model_state"])
        return self

    @torch.no_grad()
    def infer(self, test_loader):
        self.model.eval()
        all_preds, all_targets = [], []
        for X, y in test_loader:
            X = X.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device_type, enabled=self.use_amp):
                pred = self.model(X).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())
            del X, y
        return (
            np.concatenate(all_preds,   axis=0).astype(np.float32),
            np.concatenate(all_targets, axis=0).astype(np.float32),
        )