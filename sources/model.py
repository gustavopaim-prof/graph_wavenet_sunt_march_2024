# model.py — versao corrigida (v6)
#
# Historico de correcoes:
#
# v1 -> v2: bug cuSPARSE FP16/FP32 — cast manual h[b].to(adj.dtype).
#
# v2 -> v3: autocast(enabled=False) em _sparse_diff; ADJ_SCALE_NORM;
#           docstring de densidade corrigida.
#
# v3 -> v4: dois bugs criticos de dtype em WaveNetBlock.forward.
#
# v4 -> v5: adequacao ao pipeline v7 (janela 15 min) — OBSOLETO.
#
# v5 -> v6: adequacao ao pipeline v7 (janela 20 min — correcao de IC-M1/M2/M3):
#
#   [V6-A] Docstrings e comentarios atualizados para in_steps=8, F=7, freq_min=20:
#          [IC-M1] A versao v5 descrevia "Entrada: (B, T_in=4, N, F=5) janela de
#          4 x 15 min = 60 min" — obsoleto desde a migracao para freq_min=20.
#          Entrada correta: (B, T_in=8, N, F=7), janela de 8 x 20 min = 160 min.
#          Saida: (B, T_out=2, N) — t+20 min e t+40 min.
#
#   [V6-B] Comentario de campo receptivo corrigido para num_layers=3:
#          [IC-M2] A versao v5 calculava "campo_receptivo = 2^2 = 4 = in_steps"
#          (correto para num_layers=2, agora obsoleto).
#          Com num_layers=3: dilacoes [2^0=1, 2^1=2, 2^2=4].
#          Campo receptivo = 2^3 = 8 = in_steps=8. Correto.
#          O assert em __init__ permanece identico — so o comentario foi corrigido.
#
#   [V6-C] _CHECKPOINT_BLOCKS comentario atualizado para num_layers=3:
#          [IC-M3] A versao v5 dizia "com 2 layers, apenas bloco 1".
#          Com num_layers=3 (> 2), o codigo entra no else e usa {1, 2}:
#          blocos de dilation=2 (indice 1) e dilation=4 (indice 2).
#          O comportamento em runtime ja estava correto — apenas o comentario
#          estava desatualizado.
#
#   [V6-D] Log de horizontes atualizado: cfg.horizons_min = [20, 40] para freq_min=20.
#          Comentario do print em __init__ ja usava cfg.horizons_min (correto);
#          apenas o exemplo no comentario "[V5-B]" foi atualizado.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ADJ_SCALE_NORM = True


def to_sparse_csr(adj_np):
    """
    Converte adjacencia numpy (N, N) float32 para sparse_csr float32.
    Mantido em float32 — cuSPARSE exige tipos homogeneos; compatibilidade
    com AMP e gerenciada em _sparse_diff via autocast(enabled=False).
    """
    return torch.FloatTensor(adj_np).to_sparse_csr()


def _adj_scale_factor(adj_np: np.ndarray) -> float:
    """Norma-Frobenius dos valores nao-nulos de adj_np."""
    nnz_vals = adj_np[adj_np > 0]
    if len(nnz_vals) == 0:
        return 1.0
    return float(np.sqrt((nnz_vals ** 2).sum()))


class GraphConv(nn.Module):
    """
    Convolucao em grafo — 4 adjacencias: geo + topo + adapt + adapt.T
    total_in = hidden_dim x (order x num_adj + 1) = 32 x (2x4 + 1) = 288
    """

    def __init__(self, in_dim, out_dim, num_adj=4, order=2):
        super().__init__()
        self.order  = order
        self.linear = nn.Linear(in_dim * (order * num_adj + 1), out_dim)

    @staticmethod
    def _sparse_diff(adj, h, scale=1.0):
        device_type = h.device.type
        with torch.amp.autocast(device_type=device_type, enabled=False):
            h_f32  = h.float()
            result = torch.stack(
                [torch.sparse.mm(adj, h_f32[b]) for b in range(h_f32.shape[0])],
                dim=0,
            )
        if scale != 1.0:
            result = result * scale
        return result

    @staticmethod
    def _adapt_diff_ste(e1, e2, h):
        B, N, C = h.shape
        CHUNK   = 256
        with torch.no_grad():
            out_sg = torch.zeros_like(h)
            for i0 in range(0, N, CHUNK):
                i1                  = min(i0 + CHUNK, N)
                scores              = F.relu(e1[i0:i1] @ e2.T)
                a_chunk             = F.softmax(scores, dim=-1)
                out_sg[:, i0:i1, :] = torch.einsum("rn,bnc->brc", a_chunk, h)
        return h + (out_sg - h).detach()

    def forward(self, x, adjs_static, e1, e2, adj_scales=None):
        x_f32 = x.float()
        out   = [x_f32]

        for k, adj in enumerate(adjs_static):
            scale = float(adj_scales[k]) if adj_scales is not None else 1.0
            h = x_f32
            for _ in range(self.order):
                h = self._sparse_diff(adj, h, scale=scale)
                out.append(h)

        h = x_f32
        for _ in range(self.order):
            h = self._adapt_diff_ste(e1, e2, h)
            out.append(h)

        h = x_f32
        for _ in range(self.order):
            h = self._adapt_diff_ste(e2, e1, h)
            out.append(h)

        return self.linear(torch.cat(out, dim=-1))


class WaveNetBlock(nn.Module):
    """Bloco TCN causal dilatado + GraphConv + gated activation."""

    def __init__(self, hidden_dim, dilation, num_nodes, num_adj=4, dropout=0.2):
        super().__init__()
        self.dilation      = dilation
        self.tcn_filter    = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 2),
                                       dilation=(1, dilation), padding=0)
        self.tcn_gate      = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 2),
                                       dilation=(1, dilation), padding=0)
        self.gconv         = GraphConv(hidden_dim, hidden_dim, num_adj=num_adj)
        self.residual_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.skip_proj     = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.bn            = nn.BatchNorm2d(hidden_dim)
        self.dropout       = nn.Dropout(dropout)

    def forward(self, x, adjs_static, e1, e2, adj_scales=None):
        orig_dtype = x.dtype
        residual   = x

        x_pad = F.pad(x, (self.dilation, 0))
        f     = torch.tanh(self.tcn_filter(x_pad))
        g     = torch.sigmoid(self.tcn_gate(x_pad))
        h     = f * g

        B, C, N, T = h.shape

        h_t = h[:, :, :, -1].permute(0, 2, 1)
        h_t = self.gconv(h_t, adjs_static, e1, e2, adj_scales)
        h_t = h_t.to(orig_dtype)
        h_t = h_t.permute(0, 2, 1).unsqueeze(-1)
        h   = torch.cat([h[:, :, :, :-1], h_t], dim=-1) if T > 1 else h_t

        h    = self.dropout(h)
        skip = self.skip_proj(h)

        if residual.shape[-1] != h.shape[-1]:
            residual = residual[:, :, :, -h.shape[-1]:]

        out = self.bn(self.residual_proj(h) + residual)
        return out, skip


class GraphWaveNet(nn.Module):
    """
    Graph WaveNet (Wu et al., 2019) — v6 adaptado ao pipeline v7 (janela 20 min).

    [V6-A] Parametros corrigidos para freq_min=20 (IC-M1):

      Entrada : (B, T_in=8,  N, F=7)   janela de 8 x 20 min = 160 min
      Saida   : (B, T_out=2, N)        t+20 min e t+40 min

      Features (F=7): loading, hora_sin, hora_cos, dia_sin, dia_cos,
                      slot_sin, slot_cos.

    [V6-B] Campo receptivo para num_layers=3 (IC-M2):
           Dilacoes nos blocos WaveNet: [2^0=1, 2^1=2, 2^2=4].
           Campo receptivo = 2^num_layers = 2^3 = 8 = in_steps.
           Cada bloco TCN opera sobre dados reais, sem padding ocioso.
           Assert em __init__ verifica: campo_receptivo <= in_steps.

    [V6-C] Checkpoint seletivo para num_layers=3 (IC-M3):
           Com num_layers=3 (> 2): _CHECKPOINT_BLOCKS = {1, 2}.
           Bloco 1 (dilation=2) e bloco 2 (dilation=4) recebem checkpointing
           — os de maior pico de ativacoes no forward pass.
           Bloco 0 (dilation=1, menor consumo) nao usa checkpointing.
    """

    def __init__(self, cfg, adj_geo, adj_topo):
        super().__init__()
        self.cfg            = cfg
        self.adj_scale_norm = ADJ_SCALE_NORM
        N                   = cfg.num_nodes

        # [V6-B] Verificacao de campo receptivo — falha explicita se
        # num_layers for incompativel com in_steps.
        import math
        campo_receptivo = 2 ** cfg.num_layers
        if campo_receptivo > cfg.in_steps:
            recomendado = max(1, int(math.log2(cfg.in_steps)))
            raise ValueError(
                f"[GWN] Campo receptivo ({campo_receptivo}) > in_steps ({cfg.in_steps}). "
                f"Reduza num_layers para {recomendado} em config.py."
            )

        # [V6-C] Checkpoint seletivo.
        # num_layers=3: blocos 1 (dilation=2) e 2 (dilation=4) — maior pico.
        # num_layers<=2: apenas bloco 1 (dilation=2).
        # Bloco 0 (dilation=1) nunca usa checkpointing em nenhuma configuracao.
        if cfg.num_layers <= 2:
            self._CHECKPOINT_BLOCKS = frozenset({1})
        else:
            self._CHECKPOINT_BLOCKS = frozenset({1, 2})

        self.register_buffer("adj_geo",  to_sparse_csr(adj_geo),  persistent=False)
        self.register_buffer("adj_topo", to_sparse_csr(adj_topo), persistent=False)

        if self.adj_scale_norm:
            scale_geo  = 1.0 / max(_adj_scale_factor(adj_geo),  1e-8)
            scale_topo = 1.0 / max(_adj_scale_factor(adj_topo), 1e-8)
        else:
            scale_geo = scale_topo = 1.0

        self.register_buffer("scale_geo",  torch.tensor(scale_geo,  dtype=torch.float32), persistent=True)
        self.register_buffer("scale_topo", torch.tensor(scale_topo, dtype=torch.float32), persistent=True)

        self.node_emb1  = nn.Embedding(N, 10)
        self.node_emb2  = nn.Embedding(N, 10)
        self.input_proj = nn.Conv2d(cfg.in_features, cfg.hidden_dim, kernel_size=(1, 1))

        self.blocks = nn.ModuleList([
            WaveNetBlock(
                hidden_dim = cfg.hidden_dim,
                # [V6-B] Dilacoes: 2^0=1, 2^1=2, 2^2=4 para num_layers=3.
                # Campo receptivo acumulado = 2^num_layers = 8 = in_steps.
                dilation   = 2 ** (i % 8),
                num_nodes  = N,
                num_adj    = cfg.num_adj,
                dropout    = cfg.dropout,
            )
            for i in range(cfg.num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(cfg.hidden_dim, cfg.out_steps,  kernel_size=(1, 1)),
        )

        geo_nnz  = int((adj_geo  > 0).sum())
        topo_nnz = int((adj_topo > 0).sum())

        # [V6-D] horizons via cfg.horizons_min — com freq_min=20: [20, 40].
        horizons_str = ", ".join(f"t+{h}min" for h in cfg.horizons_min)

        print(
            f"[GWN v6] N={N} | F={cfg.in_features} | num_adj={cfg.num_adj} | "
            f"num_layers={cfg.num_layers}\n"
            f"  in_steps={cfg.in_steps} ({cfg.in_steps*cfg.freq_min}min) | "
            f"out_steps={cfg.out_steps} ({horizons_str})\n"
            f"  campo_receptivo={campo_receptivo} timesteps "
            f"({campo_receptivo*cfg.freq_min}min) — OK\n"
            f"  dilacoes nos blocos: "
            f"{[2**(i%8) for i in range(cfg.num_layers)]}\n"
            f"  adj_geo  : {geo_nnz:,} arestas | dens={geo_nnz/N**2:.5f}"
            f" | ~{geo_nnz*12/1e6:.0f}MB sparse"
            f" | scale={scale_geo:.4f} (ADJ_SCALE_NORM={self.adj_scale_norm})\n"
            f"  adj_topo : {topo_nnz:,} arestas | dens={topo_nnz/N**2:.6f}"
            f" | ~1MB sparse | scale={scale_topo:.4f}\n"
            f"  AMP      : autocast(False)+h.float() em sparse_mm; "
            f"h_t.to(orig_dtype) antes do cat em WaveNetBlock\n"
            f"  checkpoint seletivo: blocos {sorted(self._CHECKPOINT_BLOCKS)} "
            f"(dilation >= 2)"
        )
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  params treinaveis: {n_params:,}")

    def forward(self, x):
        """
        x : (B, T_in=8, N, F=7) -> (B, T_out=2, N)

        Permuta para (B, F, N, T_in) antes da projecao de entrada,
        processa os blocos WaveNet empilhados e projeta a soma dos
        skips no passo final para (B, out_steps, N).
        """
        x  = x.permute(0, 3, 2, 1)   # (B, F, N, T_in)
        x  = self.input_proj(x)       # (B, hidden_dim, N, T_in)

        e1 = self.node_emb1.weight    # (N, 10)
        e2 = self.node_emb2.weight    # (N, 10)

        adjs_static = [self.adj_geo, self.adj_topo]
        adj_scales  = [self.scale_geo, self.scale_topo] if self.adj_scale_norm else None

        skip_sum = 0
        for i, block in enumerate(self.blocks):
            if i in self._CHECKPOINT_BLOCKS:
                x, skip = checkpoint(
                    block, x, adjs_static, e1, e2, adj_scales,
                    use_reentrant=False,
                )
            else:
                x, skip = block(x, adjs_static, e1, e2, adj_scales)
            skip_sum = skip_sum + skip

        # skip_sum[:, :, :, -1:] : ultimo passo temporal -> (B, hidden_dim, N, 1)
        # output_proj             : (B, out_steps, N, 1)
        # squeeze(-1)             : (B, out_steps, N)
        out = self.output_proj(skip_sum[:, :, :, -1:])
        return out.squeeze(-1)