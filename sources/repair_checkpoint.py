"""
repair_checkpoint.py
====================
Injeta o campo 'node_order' em um checkpoint legado (treinado antes da
correcao [T7] de train.py) usando a ordem canonica de nos persistida em
node_metadata.csv pelo pipeline de pre-processamento.

Uso:
    python repair_checkpoint.py

    # Ou com caminhos customizados:
    python repair_checkpoint.py \
        --checkpoint checkpoints/best_model.pt \
        --metadata   training_data/node_metadata.csv \
        --train_mean training_data/train_mean.csv \
        --output     checkpoints/best_model_repaired.pt

O que o script faz:
    1. Carrega o checkpoint legado.
    2. Le node_metadata.csv e reconstroi node_order ordenado por node_index.
    3. Verifica consistencia: len(node_order) deve bater com o campo
       'num_nodes' do checkpoint (se presente) e com train_mean.csv.
    4. Injeta node_order no dict do checkpoint e salva o arquivo reparado.
    5. Loga um diff resumido entre node_order e stop_order (train_mean.csv)
       para confirmar que o alinhamento esta correto apos o reparo.

Seguranca:
    - O checkpoint original NAO e sobrescrito: o arquivo de saida e
      distinto (sufixo '_repaired' por padrao).
    - Se o checkpoint ja contem 'node_order', o script emite aviso e
      encerra sem modificar nada, a menos que --force seja passado.
"""

import argparse
import os
import sys
import logging

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)


def parse_args():
    # Detecta PROJECT_ROOT como dois niveis acima deste script
    here        = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)

    p = argparse.ArgumentParser(description="Injeta node_order em checkpoint legado.")
    p.add_argument("--checkpoint",
                   default=os.path.join(project_root, "checkpoints", "best_model.pt"),
                   help="Caminho do checkpoint a reparar.")
    p.add_argument("--metadata",
                   default=os.path.join(project_root, "training_data", "node_metadata.csv"),
                   help="Caminho do node_metadata.csv gerado pelo pipeline.")
    p.add_argument("--train_mean",
                   default=os.path.join(project_root, "training_data", "train_mean.csv"),
                   help="Caminho do train_mean.csv para validacao cruzada.")
    p.add_argument("--output",
                   default=None,
                   help="Caminho de saida. Padrao: <checkpoint>_repaired.pt")
    p.add_argument("--force",
                   action="store_true",
                   help="Sobrescreve node_order mesmo se ja existir no checkpoint.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Validacao de existencia dos arquivos ──────────────────────────────────
    for path, label in [
        (args.checkpoint,  "Checkpoint"),
        (args.metadata,    "node_metadata.csv"),
        (args.train_mean,  "train_mean.csv"),
    ]:
        if not os.path.exists(path):
            log.error(f"{label} nao encontrado: {path}")
            sys.exit(1)

    # ── Carrega checkpoint ────────────────────────────────────────────────────
    log.info(f"Carregando checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict) or "model_state" not in ckpt:
        log.error(
            "Checkpoint nao esta no formato dict com 'model_state'. "
            "Este script so suporta checkpoints gerados pelo train.py atual."
        )
        sys.exit(1)

    saved_epoch = ckpt.get("epoch", "?")
    best_val    = ckpt.get("best_val", float("nan"))
    log.info(f"  epoca={saved_epoch} | best_val_MAE={best_val:.4f}")

    # ── Verifica se node_order ja existe ─────────────────────────────────────
    if "node_order" in ckpt:
        if not args.force:
            log.warning(
                "Checkpoint ja contem 'node_order' — nenhuma modificacao necessaria. "
                "Use --force para sobrescrever."
            )
            sys.exit(0)
        else:
            log.warning("--force ativo: node_order existente sera sobrescrito.")

    # ── Le node_metadata.csv e reconstroi node_order ──────────────────────────
    log.info(f"Lendo node_metadata.csv: {args.metadata}")
    meta = pd.read_csv(args.metadata)

    col_id  = "node_id"
    col_idx = "node_index"

    if col_id not in meta.columns or col_idx not in meta.columns:
        log.error(
            f"node_metadata.csv deve conter as colunas '{col_id}' e '{col_idx}'. "
            f"Colunas encontradas: {list(meta.columns)}"
        )
        sys.exit(1)

    # Ordena por node_index para garantir a ordem canonica do tensor
    meta_sorted = meta.sort_values(col_idx).reset_index(drop=True)
    node_order  = meta_sorted[col_id].tolist()
    N_meta      = len(node_order)
    log.info(f"  {N_meta:,} nos reconstruidos de node_metadata.csv")

    # ── Valida consistencia com o checkpoint ──────────────────────────────────
    ckpt_num_nodes = ckpt.get("num_nodes")
    if ckpt_num_nodes is not None and int(ckpt_num_nodes) != N_meta:
        log.error(
            f"Inconsistencia: checkpoint diz num_nodes={ckpt_num_nodes}, "
            f"mas node_metadata.csv tem {N_meta} nos. "
            f"Verifique se o metadata pertence ao mesmo experimento."
        )
        sys.exit(1)

    # Verifica via shape do primeiro parametro do model_state
    first_param = next(iter(ckpt["model_state"].values()))
    # node embeddings: shape (N, 10) — primeira dimensao deve ser N
    node_emb_shapes = {
        k: v.shape for k, v in ckpt["model_state"].items()
        if "node_emb" in k
    }
    if node_emb_shapes:
        emb_n = list(node_emb_shapes.values())[0][0]
        if emb_n != N_meta:
            log.error(
                f"Inconsistencia critica: node embeddings do modelo tem N={emb_n}, "
                f"mas node_metadata.csv tem {N_meta} nos. "
                f"O metadata.csv nao pertence a este checkpoint."
            )
            sys.exit(1)
        log.info(
            f"  Consistencia N OK: node embeddings={emb_n} == metadata={N_meta}"
        )

    # ── Valida alinhamento com train_mean.csv ─────────────────────────────────
    log.info(f"Validando alinhamento com train_mean.csv: {args.train_mean}")
    train_mean  = pd.read_csv(args.train_mean, index_col=0)
    stop_order  = list(train_mean.index)
    N_mean      = len(stop_order)

    if N_mean != N_meta:
        log.error(
            f"train_mean.csv tem {N_mean} nos, node_metadata.csv tem {N_meta}. "
            f"Os artefatos parecem pertencer a execucoes diferentes do pipeline."
        )
        sys.exit(1)

    # Conta divergencias entre node_order reconstruido e stop_order
    n_diff = sum(a != b for a, b in zip(node_order, stop_order))
    if n_diff > 0:
        log.warning(
            f"  {n_diff:,}/{N_meta:,} posicoes divergem entre node_order "
            f"(metadata) e stop_order (train_mean). "
            f"Isso indica que train_mean.csv foi gerado com uma ordenacao "
            f"diferente da persistida em node_metadata.csv. "
            f"O node_order do checkpoint sera a ordem do metadata (canonica do tensor)."
        )
        # Mostra as primeiras 5 divergencias para diagnostico
        divergencias = [
            (i, a, b)
            for i, (a, b) in enumerate(zip(node_order, stop_order))
            if a != b
        ][:5]
        for i, a, b in divergencias:
            log.warning(f"    posicao {i}: metadata='{a}' vs train_mean='{b}'")
        if n_diff > 5:
            log.warning(f"    ... e mais {n_diff - 5} divergencias.")

        # DECISAO: node_order canônico e o do metadata (alinhado ao tensor X)
        # train_mean.csv sera reordenado em evaluate.py via .reindex(node_order)
        # que ja e o comportamento de desnormalize() — nao requer alteracao.
        log.info(
            "  DECISAO: node_order do metadata sera usado como ordem canonica "
            "(alinhada ao tensor). desnormalize() em evaluate.py usa .reindex(), "
            "portanto funciona corretamente mesmo que train_mean esteja em outra ordem."
        )
    else:
        log.info(f"  Alinhamento perfeito: node_order == stop_order ({N_meta} nos).")

    # ── Injeta node_order e salva ─────────────────────────────────────────────
    ckpt["node_order"] = node_order

    if args.output is None:
        base, ext    = os.path.splitext(args.checkpoint)
        args.output  = base + "_repaired" + ext

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(ckpt, args.output)
    log.info(f"Checkpoint reparado salvo em: {args.output}")

    # ── Resumo final ──────────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 60)
    log.info("REPARO CONCLUIDO")
    log.info(f"  Checkpoint original : {args.checkpoint}")
    log.info(f"  Checkpoint reparado : {args.output}")
    log.info(f"  node_order injetado : {N_meta:,} nos")
    log.info(f"  Divergencias c/ train_mean: {n_diff:,}")
    log.info("")
    log.info("PROXIMO PASSO:")
    log.info(f"  Atualize cfg.CHECKPOINT_NAME em config.py para apontar para")
    log.info(f"  '{os.path.basename(args.output)}', ou passe --checkpoint ao avaliar.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()