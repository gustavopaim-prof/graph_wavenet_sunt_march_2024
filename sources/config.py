# config.py — versao corrigida para pipeline v7 (janela 20 minutos)
#
# Alteracoes em relacao a versao anterior:
#
# [A1] in_steps  : 12 -> 4 -> 8
#      8 x 20 min = 160 min de historico.
#
# [A2] out_steps : 6 -> 2
#      2 x 20 min = horizontes t+20 min e t+40 min.
#      Os scripts de avaliacao calculam horizontes via cfg.horizons_min.
#
# [A3] num_layers : 4 -> 2 -> 3  (CRITICO)
#      O GWN empilha blocos com dilacoes [1, 2, 4, ...] por layer.
#      Com in_steps=8, a regra e: campo_receptivo = 2^num_layers <= in_steps.
#      Com 3 layers: campo receptivo = 2^3 = 8 = in_steps. Correto.
#      Com 4 layers: campo receptivo = 16 >> 8 — as ultimas layers
#      processam majoritariamente padding zero, nao dados reais.
#
# [A4] freq_min : atributo de janela de agregacao em minutos (20).
#      [IC-C1] Corrigido de 15 para 20 em todos os comentarios.
#      Deve coincidir com CFG.freq_min do pipeline de pre-processamento.
#      Usado por horizons_min: [(h+1)*freq_min for h in range(out_steps)]
#      = [20, 40] com freq_min=20 e out_steps=2.
#
# [A5] dropout : validacao corrigida de (0, 1) exclusivo para [0, 1).
#      dropout=0.0 e valido (sem regularizacao).
#
# [A6] scheduler_patience renomeado para lr_patience.
#      Evita ambiguidade com o atributo patience do early stopping.
#
# [A7] in_features : 5 -> 7  (CORRECAO)
#      O pipeline v7 adicionou slot_sin e slot_cos em add_temporal_features().
#      Features: loading, hora_sin, hora_cos, dia_sin, dia_cos, slot_sin, slot_cos.
#      O assert em train.py verifica X_train.shape[2] == cfg.in_features.
#
# [A8] horizon_weights : pesos por horizonte para weighted_masked_mae_multihorizon().
#      [IC-C3] Corrigido: comentarios descreviam t+15/t+30 (versao anterior,
#      freq_min=15). Com freq_min=20 e out_steps=2, os horizontes sao
#      t+20 min (h=0) e t+40 min (h=1).
#      [2.0, 1.0] prioriza t+20 min sobre t+40 min, revertendo o vies
#      de melhor desempenho no horizonte mais distante observado nos experimentos.
#      [1.0, 1.0] reproduz o comportamento anterior sem ponderacao.

import os
import math
import warnings
import multiprocessing
from dataclasses import dataclass, field


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:

    # ── Caminhos ──────────────────────────────────────────────────────────────
    CHECKPOINT_DIR : str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "checkpoints")
    )
    ARTIFACTS_DIR  : str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "training_data")
    )
    CHECKPOINT_NAME: str = "best_model.pt"

    # ── Janela temporal ───────────────────────────────────────────────────────
    # [A4] freq_min=20: frequencia da janela de agregacao em minutos.
    #      [IC-C1] Corrigido de 15 para 20 — alinhado com CFG.freq_min=20
    #      do pipeline de pre-processamento (pre_processing_pipeline.py).
    #      Horizontes resultantes via horizons_min: [20, 40] min.
    freq_min      : int = 20
    target_col_idx: int = 0    # loading — indice 0 em F=7
    in_steps      : int = 8    # [A1] 8 x 20 min = 160 min de contexto
    out_steps     : int = 2    # horizontes: 20 min / 40 min

    # ── Grafo ─────────────────────────────────────────────────────────────────
    num_nodes: int = 0         # sobrescrito em runtime por X_train.shape[1]

    # ── Arquitetura ───────────────────────────────────────────────────────────
    # [A7] in_features=7: loading, hora_sin, hora_cos, dia_sin, dia_cos,
    #      slot_sin, slot_cos.
    #      slot_sin/cos capturam a posicao absoluta dentro da semana em
    #      timesteps de 20 min, discriminando o cruzamento dia x hora
    #      ausente nas features anteriores.
    in_features: int   = 7

    hidden_dim : int   = 32

    # [A3] num_layers=3: campo receptivo = 2^3 = 8 = in_steps.
    #      Dilacoes nos blocos WaveNet: [2^0=1, 2^1=2, 2^2=4].
    #      Todas as layers operam sobre dados reais, sem padding desperdicado.
    num_layers : int   = 3

    dropout    : float = 0.2

    # num_adj=4: geo + topo + adapt + adapt.T (Wu et al. 2019).
    num_adj    : int   = 4

    # ── Treinamento ───────────────────────────────────────────────────────────
    batch_size  : int   = 8
    lr          : float = 1e-4
    weight_decay: float = 1e-4
    epochs      : int   = 150    #150 para GWN e 150 para AGCRN
    patience    : int   = 20     #20 para GWN e 20 para AGCRN
    clip_grad   : float = 5.0

    # ── Perda ponderada por demanda ───────────────────────────────────────────
    demand_weight_beta: float = 0.5

    # ── Perda ponderada por horizonte ─────────────────────────────────────────
    # [A8] horizon_weights: pesos por horizonte de predicao.
    #      Comprimento deve ser igual a out_steps (=2).
    #
    #      [IC-C3] CORRECAO: comentarios anteriores mencionavam t+15/t+30
    #      (freq_min=15). Com freq_min=20 e out_steps=2:
    #        h=0 -> t+20 min  (horizonte imediato — maior interesse operacional)
    #        h=1 -> t+40 min  (horizonte distante)
    #
    #      [2.0, 1.0] -> prioriza t+20 min (h=0) sobre t+40 min (h=1).
    #      [1.0, 1.0] -> sem ponderacao (comportamento original sem horizon_weights).
    #
    #      Motivacao: o padrao invertido observado (MAE menor em t+40 do que
    #      em t+20) sugere que a loss sem ponderacao permite ao modelo
    #      "ceder" o horizonte imediato. Peso 2x em t+20 induz pressao
    #      explicita sobre o horizonte de maior interesse operacional.
    horizon_weights: list = field(default_factory=lambda: [2.0, 1.0])

    # ── Hardware ──────────────────────────────────────────────────────────────
    device     : str  = "cuda"
    use_amp    : bool = True
    pin_memory : bool = True

    # num_workers adaptativo: min(4, cpu_count // 2).
    num_workers: int = field(
        default_factory=lambda: min(4, multiprocessing.cpu_count() // 2)
    )

    # ── Validacao ─────────────────────────────────────────────────────────────
    def __post_init__(self):
        """
        Valida a configuracao imediatamente apos a criacao.
        Falha explicita em vez de propagar erros silenciosos durante
        o treinamento.
        """
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim deve ser positivo, recebido: {self.hidden_dim}")
        if self.in_steps <= 0:
            raise ValueError(f"in_steps deve ser positivo, recebido: {self.in_steps}")
        if self.out_steps <= 0:
            raise ValueError(f"out_steps deve ser positivo, recebido: {self.out_steps}")
        if self.freq_min <= 0:
            raise ValueError(f"freq_min deve ser positivo, recebido: {self.freq_min}")

        # [A5] dropout=0.0 e valido — condicao e [0, 1).
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout deve estar em [0, 1), recebido: {self.dropout}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size deve ser positivo, recebido: {self.batch_size}")
        if self.num_adj not in (2, 4):
            raise ValueError(f"num_adj deve ser 2 ou 4, recebido: {self.num_adj}")
        if self.lr <= 0:
            raise ValueError(f"lr deve ser positivo, recebido: {self.lr}")
        if self.patience <= 0:
            raise ValueError(f"patience deve ser positivo, recebido: {self.patience}")

        # [A8] horizon_weights deve ter comprimento igual a out_steps.
        if len(self.horizon_weights) != self.out_steps:
            raise ValueError(
                f"horizon_weights tem {len(self.horizon_weights)} elemento(s), "
                f"mas out_steps={self.out_steps}. Os comprimentos devem ser iguais."
            )
        if any(w <= 0 for w in self.horizon_weights):
            raise ValueError(
                f"Todos os horizon_weights devem ser positivos, "
                f"recebido: {self.horizon_weights}"
            )

        # [A3] Aviso quando num_layers produz campo receptivo > in_steps.
        campo_receptivo = 2 ** self.num_layers
        if campo_receptivo > self.in_steps:
            recomendado = max(1, int(math.log2(self.in_steps)))
            warnings.warn(
                f"Campo receptivo do GWN ({campo_receptivo}) excede in_steps "
                f"({self.in_steps}). As ultimas layers processam padding em vez "
                f"de dados reais. Considere num_layers={recomendado}.",
                UserWarning,
                stacklevel=2,
            )

        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.ARTIFACTS_DIR,  exist_ok=True)

    # ── Propriedades derivadas ─────────────────────────────────────────────────
    @property
    def checkpoint_path(self) -> str:
        """Caminho completo do checkpoint de melhor validacao."""
        return os.path.join(self.CHECKPOINT_DIR, self.CHECKPOINT_NAME)

    @property
    def horizons_min(self) -> list:
        """
        [A4] Horizontes de predicao em minutos.
        Com freq_min=20 e out_steps=2: [20, 40].
        Formula: [(h+1)*freq_min for h in range(out_steps)].
        """
        return [(h + 1) * self.freq_min for h in range(self.out_steps)]

    @property
    def lr_patience(self) -> int:
        """
        [A6] Patience do ReduceLROnPlateau.
        Metade do patience do early stopping garante pelo menos uma
        reducao de LR antes do treino parar.
        """
        return max(1, self.patience // 2)