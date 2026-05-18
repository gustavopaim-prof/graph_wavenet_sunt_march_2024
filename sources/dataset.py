# dataset.py — versao corrigida
#
# Alteracoes em relacao a versao anterior:
#
# [D1] make_loaders aceita None para splits nao utilizados (IC-D1 — CRITICO):
#      train.py [T6] passa X_test=None (nao usa test_loader).
#      evaluate.py [E6] passa X_train=None e X_val=None (nao usa esses loaders).
#      A versao anterior criava SlidingWindowDataset(None, cfg) incondicionalmente,
#      lancando TypeError em np.ascontiguousarray(None).
#      Correcao: quando X e None, o loader correspondente retorna None.
#      O chamador verifica o retorno antes de iterar (ex: "if train_loader:").
#
# [D2] plot_results.py ainda passa todos os tres splits — sem crash apos [D1],
#      mas continua alocando X_train e X_val desnecessariamente em RAM.
#      Recomenda-se atualizar plot_results.py para passar None nos splits
#      nao utilizados (veja comentario em make_loaders abaixo).

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class SlidingWindowDataset(Dataset):
    """
    Janela deslizante sobre a serie temporal.

    Shapes retornados por __getitem__:
      x : (in_steps,  N, F)   ->  apos batch: (B, in_steps,  N, F)
      y : (out_steps, N)      ->  apos batch: (B, out_steps, N)

    Implementacao:
      - torch.from_numpy + ascontiguousarray garante zero-copy quando o
        array de entrada ja e float32 C-contiguous — evita duplicar o
        dataset inteiro em RAM (critico para N grande com T grande).
      - A conversao para float32 ocorre uma unica vez no construtor;
        cada __getitem__ retorna views do tensor compartilhado.
    """

    def __init__(self, array: np.ndarray, cfg):
        # torch.FloatTensor(array) fazia copia eager do array inteiro.
        # torch.from_numpy compartilha memoria com o array NumPy (zero-copy)
        # quando dtype=float32 e layout C-contiguous.
        array = np.ascontiguousarray(array, dtype=np.float32)
        self.tensor     = torch.from_numpy(array)
        self.in_steps   = cfg.in_steps
        self.out_steps  = cfg.out_steps
        self.target_idx = cfg.target_col_idx

    def __len__(self) -> int:
        # Numero de janelas validas: cada janela precisa de in_steps + out_steps
        # timesteps contiguos. Se o tensor for curto demais, retorna 0 (seguro).
        return max(0, self.tensor.shape[0] - self.in_steps - self.out_steps + 1)

    def __getitem__(self, idx: int):
        x_start = idx
        x_end   = idx + self.in_steps
        y_start = x_end
        y_end   = x_end + self.out_steps

        x = self.tensor[x_start:x_end]                          # (in_steps, N, F)
        y = self.tensor[y_start:y_end, :, self.target_idx]      # (out_steps, N)
        return x, y


def make_loaders(
    X_train: Optional[np.ndarray],
    X_val:   Optional[np.ndarray],
    X_test:  Optional[np.ndarray],
    cfg,
):
    """
    Cria DataLoaders para treino, validacao e teste.

    Retorna uma tupla (train_loader, val_loader, test_loader).
    Quando um split e None, o loader correspondente e None.

    [D1] CORRECAO IC-D1: splits None sao suportados.
    Chamadores que nao usam todos os splits devem passar None para
    evitar alocacao desnecessaria de DataLoaders e consumo de RAM:

      # train.py — nao usa test_loader:
      train_loader, val_loader, _ = make_loaders(X_train, X_val, None, cfg)

      # evaluate.py — nao usa train_loader nem val_loader:
      _, _, test_loader = make_loaders(None, None, X_test, cfg)

      # plot_results.py — idem evaluate.py (recomendado apos [D2]):
      _, _, test_loader = make_loaders(None, None, X_test, cfg)

    Decisoes de design dos DataLoaders ativos:
      - drop_last=True no treino: evita batch incompleto no fim da epoca.
        BatchNorm com B=1 tem variancia indefinida e degrada o gradiente.
        Validacao e teste usam drop_last=False para avaliar todos os samples.

      - prefetch_factor=2: com num_workers > 0, o proximo batch e pre-carregado
        em RAM enquanto a GPU processa o batch atual.

      - persistent_workers=True: mantem os processos worker vivos entre epocas,
        evitando overhead de fork/spawn a cada epoca.

      - pin_memory so e ativado se CUDA estiver disponivel — em CPU-only,
        pin_memory nao tem efeito e pode causar aviso em algumas versoes do PyTorch.
    """
    num_workers = getattr(cfg, "num_workers", 0)
    pin_memory  = getattr(cfg, "pin_memory", False) and torch.cuda.is_available()

    # prefetch_factor so e valido quando num_workers > 0.
    # Passar prefetch_factor com num_workers=0 lanca ValueError no PyTorch >= 1.12.
    prefetch = 2 if num_workers > 0 else None

    base_kwargs = dict(
        batch_size         = cfg.batch_size,
        num_workers        = num_workers,
        pin_memory         = pin_memory,
        persistent_workers = num_workers > 0,
        prefetch_factor    = prefetch,
    )

    # [D1] Guard: se o split e None, retorna None sem tentar construir o Dataset.
    def _make(X, shuffle, drop_last):
        if X is None:
            return None
        return DataLoader(
            SlidingWindowDataset(X, cfg),
            shuffle   = shuffle,
            drop_last = drop_last,
            **base_kwargs,
        )

    # drop_last=True no treino: evita batch de B=1 com BatchNorm instavel.
    # Validacao e teste: drop_last=False para nao perder samples na avaliacao.
    train_loader = _make(X_train, shuffle=True,  drop_last=True)
    val_loader   = _make(X_val,   shuffle=False, drop_last=False)
    test_loader  = _make(X_test,  shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader