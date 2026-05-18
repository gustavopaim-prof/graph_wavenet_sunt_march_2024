# pre_processing_pipeline.py — pipeline
#
# Alteracoes em relacao a versao anterior:
#
# [P1] slot_sin/cos ja eram calculados mas nao persistidos — corrigido.
#
# [P2] ALL_FEATURES atualizado para incluir slot_sin e slot_cos (F=7).
#
# [P3] Assert atualizado: F=7 apos adicao de slot_sin e slot_cos.
#
# [P4] Log final corrigido: valores derivados do tensor gerado.
#
# [P5] check_temporal_equidistance corrigido (IC-6):
#      A versao anterior usava comparacao exata (diffs_min != freq_min),
#      sensivel a offsets de fuso horario (DST) que introduzem deltas de
#      +/-1 segundo em timestamps UTC ao cruzar mudancas de horario de verao.
#      Tolerancia de +/-1 segundo (1/60 min) introduzida.
#
# [P6] CORRECAO CRITICA — divergencia de ordem entre node_order e train_mean:
#
#      CAUSA RAIZ:
#        ValueError: Desalinhamento entre node_order do checkpoint e
#        stop_order de train_mean.csv: 5061 posicoes divergem (N=18340).
#
#      A Etapa 8 derivava node_order via drop_duplicates(), que preserva a
#      primeira ocorrencia de cada node_id na ordem do DataFrame pos-split.
#      Essa ordem NAO E DETERMINISTICA: varia conforme versao do pandas,
#      hash de strings e ordem de leitura dos arquivos Parquet.
#
#      A Etapa 10 calculava train_mean via groupby("node_id").mean(), que
#      SEMPRE ordena o indice ALFABETICAMENTE (sort=True e o padrao do
#      pandas GroupBy). Com N=18340 nos, a ordem de chegada e a ordem
#      alfabetica divergem em ~5000 posicoes — exatamente o observado.
#
#      CORRECAO em dois pontos:
#
#      (a) Etapa 8 — node_order forcado a ser ordenado ALFABETICAMENTE:
#          nodes_unique.sort_values("node_id") apos drop_duplicates().
#          Isso torna node_order identico ao indice que groupby() produz,
#          eliminando a divergencia pela raiz. A ordenacao e deterministica
#          em qualquer versao do pandas.
#
#      (b) Etapa 10 — train_mean e train_std reindexados por node_order:
#          Como salvaguarda adicional e para tornar o invariante auditavel,
#          ambos os DataFrames recebem .reindex(node_order) apos o groupby.
#          Com a correcao (a), isso e uma operacao identidade, mas garante
#          alinhamento mesmo se o comportamento do pandas mudar.
#          Um assert verifica list(train_mean.index) == node_order antes
#          de prosseguir.
#
# [P7] min_obs_frac: 0.05 -> 0.25
#      MOTIVACAO: com min_obs_frac=0.05 (5%), paradas com ate 95% de
#      imputacao por ffill passavam pelo filtro. O grafico de serie
#      temporal revelou paradas com apenas 22% de dados reais — o modelo
#      aprendia sobre a constante imputada, nao sobre demanda genuina.
#      Metricas calculadas sobre esses nos inflavam o erro global e
#      obscureciam o desempenho real do modelo.
#
#      CALCULO: com T_treino~1452 timesteps (30.9 dias, freq=20min):
#        0.05 -> min. 73 obs  (~24h) — exclui apenas paradas fantasma
#        0.25 -> min. 363 obs (~121h / 5 dias) — exclui paradas com
#               menos de 25% de dado real no periodo de treino.
#
#      Uma parada com 22% de dados reais (320 obs) tem 320 < 363:
#      agora e corretamente excluida.
#
#      IMPACTO ESPERADO: reducao de N de ~18.406 para ~12.000-15.000 nos
#      (estimativa; valor exato gerado pelo pipeline). Metricas mais
#      honestas — refletem desempenho em paradas operacionalmente ativas.
#      Aplicado identicamente a GWN e AGCRN para manter comparacao justa.

import gc
import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# ============================================================
# LOGGING
# ============================================================
_stream_handler        = logging.StreamHandler(sys.stdout)
_stream_handler.stream = open(sys.stdout.fileno(),
                               mode="w", encoding="utf-8",
                               closefd=False, buffering=1)
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s | %(levelname)s | %(message)s",
    handlers = [
        logging.FileHandler("preprocessing.log", encoding="utf-8"),
        _stream_handler,
    ],
)
log = logging.getLogger(__name__)


# ============================================================
# CONFIGURACAO
# ============================================================
@dataclass
class PreprocessingConfig:
    """Hiperparametros do pipeline. Mesmos valores -> mesmo output (hash MD5)."""
    project_root : str = field(default_factory=lambda:
                                os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    source_dir   : str = ""
    gtfs_dir     : str = ""
    artifacts_dir: str = ""

    freq_min     : int   = 20
    min_obs_frac : float = 0.25   # [P7] elevado de 0.05 para excluir paradas majoritariamente imputadas
    geo_sigma    : float = 0.5
    geo_threshold: float = 0.1
    geo_chunk    : int   = 500
    split_train  : float = 0.70
    split_val    : float = 0.85
    tz_local     : str   = "America/Bahia"

    def __post_init__(self):
        root = self.project_root
        if not self.source_dir:
            self.source_dir    = os.path.join(root, "sunt_march_2024")
        if not self.gtfs_dir:
            self.gtfs_dir      = os.path.join(root, "sunt_coordinations")
        if not self.artifacts_dir:
            self.artifacts_dir = os.path.join(root, "training_data")
        self.tz_local = os.environ.get("TZ_LOCAL", self.tz_local)
        os.makedirs(self.artifacts_dir, exist_ok=True)

    @property
    def freq_str(self) -> str:
        return f"{self.freq_min}min"

    @property
    def minutos_por_dia(self) -> int:
        return 1440 // self.freq_min


CFG = PreprocessingConfig()

log.info(f"Raiz do projeto : {CFG.project_root}")
log.info(f"Dados fonte     : {CFG.source_dir}")
log.info(f"GTFS            : {CFG.gtfs_dir}")
log.info(f"Artefatos       : {CFG.artifacts_dir}")
log.info(f"Fuso local      : {CFG.tz_local}")
log.info(f"Janela          : {CFG.freq_min} min ({CFG.minutos_por_dia} timesteps/dia)")
log.info(f"min_obs_frac    : {CFG.min_obs_frac:.0%} (referenciado ao treino)")
log.info(
    f"Config modelos  : in_steps=8 (8x{CFG.freq_min}min={8*CFG.freq_min}min) | "
    f"out_steps=2 (t+{CFG.freq_min}min, t+{CFG.freq_min*2}min)"
)


# ============================================================
# UTILITARIOS
# ============================================================

def clean_id(x) -> str:
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()


def parse_direction_id(series: pd.Series) -> pd.Series:
    DIR_MAP = {"I": 0, "V": 1}
    s       = series.astype(str).str.strip().str.upper()
    mapped  = s.map(DIR_MAP)
    numeric = pd.to_numeric(s, errors="coerce")
    result  = mapped.combine_first(numeric).fillna(-1).astype(int)
    counts  = series.astype(str).value_counts()
    n_unk   = int((result == -1).sum())
    log.info(
        f"  direction_id - valores: {counts.to_dict()}"
        + (f" | desconhecidos->-1: {n_unk:,}" if n_unk else "")
    )
    return result


def sym_norm(adj: np.ndarray) -> np.ndarray:
    deg        = adj.sum(axis=1)
    safe_deg   = np.where(deg > 0, deg, 1.0)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(safe_deg), 0.0).astype(np.float32)
    return (d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]).astype(np.float32)


def md5_short(arr: np.ndarray) -> str:
    return hashlib.md5(arr.tobytes()).hexdigest()[:8]


def save_artifact(arr: np.ndarray, path: str, label: str) -> None:
    np.save(path, arr)
    log.info(f"  {label}: shape={arr.shape} | md5={md5_short(arr)}")


def log_imputation_stats(df: pd.DataFrame, split_name: str) -> None:
    n_null  = df["loading"].isna().sum()
    n_total = len(df)
    log.info(
        f"[{split_name}] Pre-imputacao: {n_null:,}/{n_total:,} "
        f"({n_null/max(n_total,1):.1%}) valores de loading eram nulos."
    )


def check_temporal_equidistance(df: pd.DataFrame, freq_min: int,
                                 split_name: str) -> None:
    # [P5] Tolerancia de +/-1s para offsets de fuso/DST.
    ts = np.sort(df["stop_time"].unique())
    if len(ts) < 2:
        return
    diffs_min = np.diff(ts).astype("timedelta64[s]").astype(float) / 60
    tol       = 1.0 / 60.0
    n_irreg   = int((np.abs(diffs_min - freq_min) > tol).sum())
    if n_irreg > 0:
        irreg_vals = np.unique(np.round(diffs_min[np.abs(diffs_min - freq_min) > tol], 2))
        log.warning(
            f"[{split_name}] {n_irreg} intervalos irregulares (tolerancia={tol*60:.0f}s). "
            f"Valores distintos encontrados (min): {irreg_vals[:10].tolist()}"
            + (" ..." if len(irreg_vals) > 10 else "")
        )
    else:
        log.info(
            f"[{split_name}] Continuidade temporal OK — "
            f"todos os intervalos = {freq_min} min (tolerancia={tol*60:.0f}s)."
        )


def add_temporal_features(df: pd.DataFrame, tz: str, freq_min: int) -> pd.DataFrame:
    """
    Adiciona features temporais ciclicas. [P1] slot_sin/cos persistidos.
    hora_sin/cos : minuto_do_dia / 1440
    dia_sin/cos  : weekday / 7
    slot_sin/cos : slot_semana / 10080  (posicao absoluta na semana)
    Todas em [-1, 1].
    """
    df = df.copy()
    try:
        t_local = df["stop_time"].dt.tz_convert(tz)
    except Exception:
        log.warning(f"Fuso '{tz}' invalido — usando UTC.")
        t_local = df["stop_time"]

    minuto_do_dia = (t_local.dt.hour * 60 + t_local.dt.minute).astype(np.float32)

    df["hora_sin"] = np.sin(2 * np.pi * minuto_do_dia / 1440).astype(np.float32)
    df["hora_cos"] = np.cos(2 * np.pi * minuto_do_dia / 1440).astype(np.float32)
    df["dia_sin"]  = np.sin(2 * np.pi * t_local.dt.weekday / 7).astype(np.float32)
    df["dia_cos"]  = np.cos(2 * np.pi * t_local.dt.weekday / 7).astype(np.float32)

    slot_semana    = (t_local.dt.weekday * 1440 + minuto_do_dia).astype(np.float32)
    df["slot_sin"] = np.sin(2 * np.pi * slot_semana / (7 * 1440)).astype(np.float32)
    df["slot_cos"] = np.cos(2 * np.pi * slot_semana / (7 * 1440)).astype(np.float32)

    log.info(
        f"  Features temporais: {df['stop_time'].nunique():,} timestamps | "
        f"resolucao: {1440 // freq_min} valores/dia | "
        f"features: hora_sin, hora_cos, dia_sin, dia_cos, slot_sin, slot_cos"
    )
    return df


# ============================================================
# 1. LEITURA DOS PARQUET
# ============================================================
REQUIRED_COLS = ["stop_id", "stop_time", "route_short_name", "direction_id", "loading"]

files = sorted([f for f in os.listdir(CFG.source_dir) if f.endswith(".parquet")])
log.info(f"Arquivos encontrados: {len(files)}")

dataset   = ds.dataset(CFG.source_dir, format="parquet")
available = set(dataset.schema.names)
missing   = set(REQUIRED_COLS) - available
if missing:
    raise ValueError(f"Colunas ausentes no Parquet: {missing}")

df_all = dataset.to_table(columns=REQUIRED_COLS).to_pandas()
df_all["stop_time"] = pd.to_datetime(df_all["stop_time"], utc=True, errors="coerce")

n_nat = df_all["stop_time"].isna().sum()
if n_nat > 0:
    log.warning(f"  {n_nat:,} registros com stop_time invalido — descartados.")
    df_all = df_all.dropna(subset=["stop_time"])

log.info(f"Shape total bruto: {df_all.shape}")


# ============================================================
# 2. LIMPEZA DE IDs
# ============================================================
df_all["stop_id"]          = df_all["stop_id"].apply(clean_id)
df_all["route_short_name"] = df_all["route_short_name"].fillna("UNKNOWN").astype(str).str.strip()
log.info("[direction_id] parquet:")
df_all["direction_id"] = parse_direction_id(df_all["direction_id"])


# ============================================================
# 3. AGREGACAO POR NO COMPOSTO — janelas de freq_min minutos
# ============================================================
log.info(f"Agregando em janelas de {CFG.freq_min} min (aggfunc=sum)...")
df_agg = (
    df_all
    .groupby(
        ["stop_id", "route_short_name", "direction_id",
         pd.Grouper(key="stop_time", freq=CFG.freq_str)],
        observed=True,
    )
    .agg({"loading": "sum"})
    .reset_index()
)
del df_all
gc.collect()

df_agg["loading"] = df_agg["loading"].replace(0, np.nan)
df_agg["node_id"] = (
    df_agg["stop_id"] + "__"
    + df_agg["route_short_name"] + "__"
    + df_agg["direction_id"].astype(str)
)

n_dias_estimado = (
    df_agg["stop_time"].max() - df_agg["stop_time"].min()
).total_seconds() / 86400

log.info(f"Nos compostos unicos (bruto): {df_agg['node_id'].nunique():,}")
log.info(
    f"Janelas geradas por no: ~{df_agg.groupby('node_id').size().mean():.0f} "
    f"(esperado: ~{CFG.minutos_por_dia * n_dias_estimado:.0f} "
    f"para {n_dias_estimado:.1f} dias)"
)


# ============================================================
# 4. MERGE GTFS — coordenadas geograficas
# ============================================================
df_stops = pd.read_csv(
    os.path.join(CFG.gtfs_dir, "stops.txt"),
    usecols=["stop_id", "stop_lat", "stop_lon"],
)
df_stops["stop_id"] = df_stops["stop_id"].apply(clean_id)
df_stops = df_stops.drop_duplicates(subset="stop_id")

df_agg = df_agg.merge(df_stops, on="stop_id", how="left")

missing_geo = df_agg[df_agg["stop_lat"].isna()]["node_id"].nunique()
if missing_geo > 0:
    log.warning(f"Nos sem coordenadas: {missing_geo:,} — removidos.")
    df_agg = df_agg.dropna(subset=["stop_lat", "stop_lon"])


# ============================================================
# 5. SPLIT TEMPORAL
# ============================================================
all_ts = np.sort(df_agg["stop_time"].unique())
n_ts   = len(all_ts)
cut1   = int(n_ts * CFG.split_train)
cut2   = int(n_ts * CFG.split_val)

ts_cut1 = all_ts[cut1]
ts_cut2 = all_ts[cut2]

df_train = df_agg[df_agg["stop_time"] <  ts_cut1].copy()
df_val   = df_agg[(df_agg["stop_time"] >= ts_cut1) & (df_agg["stop_time"] < ts_cut2)].copy()
df_test  = df_agg[df_agg["stop_time"] >= ts_cut2].copy()
del df_agg
gc.collect()

n_ts_treino = df_train["stop_time"].nunique()
n_ts_val    = df_val["stop_time"].nunique()
n_ts_teste  = df_test["stop_time"].nunique()

log.info(
    f"Timesteps - treino: {n_ts_treino:,} ({n_ts_treino*CFG.freq_min/60:.1f}h) | "
    f"val: {n_ts_val:,} ({n_ts_val*CFG.freq_min/60:.1f}h) | "
    f"teste: {n_ts_teste:,} ({n_ts_teste*CFG.freq_min/60:.1f}h)"
)

assert df_train["stop_time"].max() < df_val["stop_time"].min(), \
    "ERRO: overlap treino/val detectado."
assert df_val["stop_time"].max() < df_test["stop_time"].min(), \
    "ERRO: overlap val/teste detectado."

check_temporal_equidistance(df_train, CFG.freq_min, "treino")
check_temporal_equidistance(df_val,   CFG.freq_min, "val")
check_temporal_equidistance(df_test,  CFG.freq_min, "teste")


# ============================================================
# 6. FILTRAGEM DE NOS COM COBERTURA INSUFICIENTE
# ============================================================
min_obs     = int(np.ceil(CFG.min_obs_frac * n_ts_treino))
obs_treino  = df_train.groupby("node_id")["loading"].count()
nodes_valid = obs_treino[obs_treino >= min_obs].index
n_before    = len(obs_treino)

df_train = df_train[df_train["node_id"].isin(nodes_valid)].copy()
df_val   = df_val[df_val["node_id"].isin(nodes_valid)].copy()
df_test  = df_test[df_test["node_id"].isin(nodes_valid)].copy()
n_after  = df_train["node_id"].nunique()

log.info(
    f"Filtragem por cobertura minima:\n"
    f"  Base                     : {n_ts_treino:,} timesteps de TREINO\n"
    f"  Threshold (min_obs_frac) : {CFG.min_obs_frac:.0%} -> min. {min_obs:,} obs.\n"
    f"  Nos antes                : {n_before:,}\n"
    f"  Nos apos                 : {n_after:,}  (-{n_before - n_after:,} removidos)"
)


# ============================================================
# 7. FEATURES TEMPORAIS — por split, apos filtragem
# ============================================================
log.info("Calculando features temporais...")
df_train = add_temporal_features(df_train, CFG.tz_local, CFG.freq_min)
df_val   = add_temporal_features(df_val,   CFG.tz_local, CFG.freq_min)
df_test  = add_temporal_features(df_test,  CFG.tz_local, CFG.freq_min)


# ============================================================
# 8. ORDEM DOS NOS DO GRAFO
#
# [P6] CORRECAO: nodes_unique ordenado ALFABETICAMENTE por node_id
# via sort_values("node_id") antes de derivar node_order.
#
# Versao anterior: drop_duplicates() preservava a primeira ocorrencia
# de cada node_id na ordem do DataFrame pos-split — ordem de chegada
# dos dados AVL, nao deterministica entre execucoes do pandas.
#
# groupby("node_id") na Etapa 10 SEMPRE produz indice alfabetico
# (sort=True e o padrao). Com node_order nao-alfabetico, os dois
# divergiam em ~5000 posicoes com N=18340 (erro observado em producao).
#
# Com sort_values("node_id"), node_order e identico ao indice do
# groupby em qualquer versao do pandas, eliminando a divergencia.
# ============================================================
nodes_unique = (
    df_train[["node_id", "stop_id", "route_short_name", "direction_id"]]
    .drop_duplicates("node_id")
    # [P6] Ordena alfabeticamente — mesma ordem que groupby() usara na Etapa 10.
    .sort_values("node_id")
    .reset_index(drop=True)
    .merge(df_stops, on="stop_id", how="left")
)

node_order = nodes_unique["node_id"].tolist()
node_index = {nid: i for i, nid in enumerate(node_order)}
N          = len(node_order)
log.info(f"Total de nos no grafo (apos filtragem): {N:,}")
log.info(f"  Ordenacao: alfabetica por node_id (deterministica, alinhada com groupby)")


# ============================================================
# 9. IMPUTACAO — por split separado
# ============================================================
def impute(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Imputa loading por no via ffill -> bfill -> 0."""
    log_imputation_stats(df, split_name)
    df = df.copy()
    df = df.sort_values(["node_id", "stop_time"])
    df["loading"] = (
        df.groupby("node_id")["loading"]
        .transform(lambda s: s.ffill().bfill())
        .fillna(0)
    )
    return df

df_train = impute(df_train, "treino")
df_val   = impute(df_val,   "val")
df_test  = impute(df_test,  "teste")


# ============================================================
# 10. NORMALIZACAO Z-SCORE — apenas loading
#
# [P6] train_mean e train_std reindexados por node_order como
# salvaguarda explicita. Com a correcao da Etapa 8, groupby() ja
# produz o mesmo indice alfabetico de node_order (operacao identidade),
# mas o reindex torna o invariante auditavel e garante alinhamento
# mesmo se o comportamento interno do pandas mudar em versoes futuras.
# Um assert verifica list(train_mean.index) == node_order antes de salvar.
# ============================================================
train_mean = (
    df_train.groupby("node_id")[["loading"]]
    .mean()
    .reindex(node_order)          # [P6] salvaguarda: forca ordem identica ao tensor
)
train_std = (
    df_train.groupby("node_id")[["loading"]]
    .std(ddof=0)
    .reindex(node_order)          # [P6] idem
    .replace(0.0, 1.0)
)

# [P6] Assert de alinhamento — falha explicita se a invariante for violada.
assert list(train_mean.index) == node_order, (
    "INVARIANTE VIOLADA: train_mean.index != node_order apos reindex. "
    "Isso indica que ha node_ids em node_order ausentes em train_mean — "
    "verifique a consistencia das Etapas 6 e 8."
)
assert list(train_std.index) == node_order, (
    "INVARIANTE VIOLADA: train_std.index != node_order apos reindex."
)
log.info(
    "Normalizacao z-score: train_mean e train_std alinhados com node_order — OK"
)


def normalize(df: pd.DataFrame, mean: pd.DataFrame, std: pd.DataFrame) -> pd.DataFrame:
    """Z-score com estatisticas do treino. Nos ausentes recebem identidade."""
    df = df.copy()
    m  = df["node_id"].map(mean["loading"]).fillna(0.0)
    s  = df["node_id"].map(std["loading"]).fillna(1.0)
    df["loading"] = ((df["loading"] - m) / s).astype(np.float32)
    return df

df_train = normalize(df_train, train_mean, train_std)
df_val   = normalize(df_val,   train_mean, train_std)
df_test  = normalize(df_test,  train_mean, train_std)


# ============================================================
# 11. TENSOR (T, N, F=7) via pivot
#
# [P2] ALL_FEATURES com slot_sin e slot_cos.
# [0] loading  [1] hora_sin  [2] hora_cos  [3] dia_sin
# [4] dia_cos  [5] slot_sin  [6] slot_cos
# ============================================================
ALL_FEATURES = [
    "loading",
    "hora_sin", "hora_cos",
    "dia_sin",  "dia_cos",
    "slot_sin", "slot_cos",
]


def _check_no_duplicates(df: pd.DataFrame, split_name: str) -> None:
    n_dup = df.duplicated(subset=["stop_time", "node_id"]).sum()
    if n_dup > 0:
        raise ValueError(f"[{split_name}] {n_dup:,} pares (stop_time, node_id) duplicados.")


def build_tensor(df: pd.DataFrame, node_order: list,
                 features: list, split_name: str = "") -> np.ndarray:
    """Constroi tensor (T, N, F) via pivot por feature."""
    _check_no_duplicates(df, split_name)

    n_miss = (
        df.pivot(index="stop_time", columns="node_id", values="loading")
        .reindex(columns=node_order).isna().sum().sum()
    )
    if n_miss > 0:
        n_total = df["stop_time"].nunique() * len(node_order)
        log.warning(
            f"[{split_name}] {n_miss:,}/{n_total:,} ({n_miss/max(n_total,1):.1%}) "
            f"celulas ausentes -> preenchidas com 0.0 (media do no normalizado)."
        )

    arrays = []
    for col in features:
        piv = (
            df.pivot(index="stop_time", columns="node_id", values=col)
              .reindex(columns=node_order)
              .fillna(0.0)
        )
        arrays.append(piv.values.astype(np.float32))
    return np.stack(arrays, axis=-1)


log.info("Construindo tensores...")
X_train = build_tensor(df_train, node_order, ALL_FEATURES, "treino")
del df_train; gc.collect()
X_val   = build_tensor(df_val,   node_order, ALL_FEATURES, "val")
del df_val;   gc.collect()
X_test  = build_tensor(df_test,  node_order, ALL_FEATURES, "teste")
del df_test;  gc.collect()

log.info(f"Tensor treino : {X_train.shape}  -> (timesteps, nos, features)")
log.info(f"Tensor val    : {X_val.shape}")
log.info(f"Tensor teste  : {X_test.shape}")

# [P3]
assert X_train.shape[2] == len(ALL_FEATURES), (
    f"F esperado={len(ALL_FEATURES)} (ALL_FEATURES={ALL_FEATURES}), "
    f"encontrado={X_train.shape[2]}"
)
assert X_train.shape[1] == N, f"N esperado={N}, encontrado={X_train.shape[1]}"

log.info(
    f"Cobertura temporal - treino: {X_train.shape[0]*CFG.freq_min/60:.1f}h | "
    f"val: {X_val.shape[0]*CFG.freq_min/60:.1f}h | "
    f"teste: {X_test.shape[0]*CFG.freq_min/60:.1f}h"
)


# ============================================================
# 12. MATRIZES DE ADJACENCIA
# ============================================================
coords     = nodes_unique[["stop_lat", "stop_lon"]].values.astype(np.float32)
coords_rad = np.radians(coords)
stop_ids_n = nodes_unique["stop_id"].to_numpy(dtype=str)

# ── 12a. Adjacencia geografica ────────────────────────────────────────────
adj_geo = np.zeros((N, N), dtype=np.float32)
log.info(f"Calculando adj_geo em chunks de {CFG.geo_chunk}...")

for i0 in range(0, N, CFG.geo_chunk):
    i1    = min(i0 + CFG.geo_chunk, N)
    lat_i = coords_rad[i0:i1, 0:1].astype(np.float32)
    lon_i = coords_rad[i0:i1, 1:2].astype(np.float32)

    dlat = (lat_i - coords_rad[:, 0]).astype(np.float32)
    dlon = (lon_i - coords_rad[:, 1]).astype(np.float32)

    sin_dlat  = np.sin(dlat / 2, dtype=np.float32)
    sin_dlon  = np.sin(dlon / 2, dtype=np.float32)
    cos_lat_i = np.cos(lat_i, dtype=np.float32)
    cos_lat_j = np.cos(coords_rad[:, 0], dtype=np.float32)

    a    = sin_dlat**2 + cos_lat_i * cos_lat_j * sin_dlon**2
    dist = (6371.0 * 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))).astype(np.float32)
    w    = np.exp(-(dist**2) / (2.0 * CFG.geo_sigma**2), dtype=np.float32)

    same_stop                = (stop_ids_n[i0:i1, None] == stop_ids_n[None, :])
    w[same_stop]             = 0.0
    w[w < CFG.geo_threshold] = 0.0
    adj_geo[i0:i1]           = w

np.fill_diagonal(adj_geo, 0.0)
log.info(f"[Geo]  Arestas brutas ativas: {int((adj_geo > 0).sum()):,}")

# ── 12b. Adjacencia topologica / GTFS ────────────────────────────────────
adj_topo = np.zeros((N, N), dtype=np.float32)

stop_times_path = os.path.join(CFG.gtfs_dir, "stop_times.txt")
trips_path      = os.path.join(CFG.gtfs_dir, "trips.txt")
routes_path     = os.path.join(CFG.gtfs_dir, "routes.txt")

if not os.path.exists(stop_times_path):
    log.warning("[Topo] stop_times.txt ausente — adj_topo zerada.")
elif not (os.path.exists(trips_path) and os.path.exists(routes_path)):
    log.warning("[Topo] trips.txt ou routes.txt ausentes — adj_topo zerada.")
else:
    df_st = pd.read_csv(
        stop_times_path, low_memory=False,
        usecols=["trip_id", "stop_id", "stop_sequence"],
        dtype={"stop_sequence": np.int32},
    )
    df_st["stop_id"] = df_st["stop_id"].apply(clean_id)

    df_trips  = pd.read_csv(trips_path)
    df_routes = pd.read_csv(routes_path)

    if "route_short_name" not in df_trips.columns:
        df_trips = df_trips.merge(
            df_routes[["route_id", "route_short_name"]], on="route_id", how="left"
        )

    df_st = df_st.merge(
        df_trips[["trip_id", "route_short_name", "direction_id"]],
        on="trip_id", how="left",
    )
    df_st["route_short_name"] = (
        df_st["route_short_name"].fillna("UNKNOWN").astype(str).str.strip()
    )
    log.info("[direction_id] GTFS stop_times:")
    df_st["direction_id"] = parse_direction_id(df_st["direction_id"])
    df_st = df_st.sort_values(["trip_id", "stop_sequence"])

    df_st["next_stop_id"] = df_st.groupby("trip_id")["stop_id"].shift(-1)
    df_st["next_trip_id"] = df_st.groupby("trip_id")["trip_id"].shift(-1)

    pairs = df_st[df_st["trip_id"] == df_st["next_trip_id"]].copy()
    pairs = pairs.dropna(subset=["next_stop_id"])

    pairs["node_a"] = (
        pairs["stop_id"].astype(str) + "__"
        + pairs["route_short_name"].astype(str) + "__"
        + pairs["direction_id"].astype(str)
    )
    pairs["node_b"] = (
        pairs["next_stop_id"].astype(str) + "__"
        + pairs["route_short_name"].astype(str) + "__"
        + pairs["direction_id"].astype(str)
    )

    pairs = pairs[
        pairs["node_a"].isin(node_index) & pairs["node_b"].isin(node_index)
    ][["node_a", "node_b"]].drop_duplicates()

    idx_a = pairs["node_a"].map(node_index).values
    idx_b = pairs["node_b"].map(node_index).values
    adj_topo[idx_a, idx_b] = 1.0
    adj_topo[idx_b, idx_a] = 1.0
    log.info(f"[Topo] Arestas brutas ativas: {int((adj_topo > 0).sum()):,}")

adj_geo_norm  = sym_norm(adj_geo)
adj_topo_norm = sym_norm(adj_topo)

log.info(f"Paradas no grafo : {N:,}")
log.info(f"Densidade geo    : {float((adj_geo_norm  > 0).mean()):.4f}")
log.info(f"Densidade topo   : {float((adj_topo_norm > 0).mean()):.4f}")


# ============================================================
# 13. SALVAR ARTEFATOS
# ============================================================
log.info("Salvando artefatos:")
save_artifact(X_train,       os.path.join(CFG.artifacts_dir, "X_train.npy"),  "X_train")
save_artifact(X_val,         os.path.join(CFG.artifacts_dir, "X_val.npy"),    "X_val")
save_artifact(X_test,        os.path.join(CFG.artifacts_dir, "X_test.npy"),   "X_test")
save_artifact(adj_geo_norm,  os.path.join(CFG.artifacts_dir, "adj_geo.npy"),  "adj_geo")
save_artifact(adj_topo_norm, os.path.join(CFG.artifacts_dir, "adj_topo.npy"), "adj_topo")

train_mean.to_csv(os.path.join(CFG.artifacts_dir, "train_mean.csv"))
train_std.to_csv( os.path.join(CFG.artifacts_dir, "train_std.csv"))

# [P6] nodes_unique ja esta ordenado por node_id (Etapa 8).
nodes_unique["node_index"] = nodes_unique["node_id"].map(node_index)
nodes_unique.to_csv(os.path.join(CFG.artifacts_dir, "node_metadata.csv"), index=False)

# [P4] Log final.
in_features_gerado = X_train.shape[2]
log.info(f"\nArtefatos salvos em  : {CFG.artifacts_dir}")
log.info(f"  Janela             : {CFG.freq_min} min | N={N:,} nos")
log.info(f"  Timesteps treino   : {X_train.shape[0]:,} ({X_train.shape[0]*CFG.freq_min/60:.1f}h)")
log.info(f"  min_obs_frac       : {CFG.min_obs_frac:.0%} (sobre treino)")
log.info(f"  Features (F)       : {in_features_gerado} {ALL_FEATURES}")
log.info(f"  hora_sin/cos       : minuto_do_dia/1440 ({1440//CFG.freq_min} valores/dia)")
log.info(f"  slot_sin/cos       : slot_semana/(7*1440) ({7*(1440//CFG.freq_min)} valores/semana)")
log.info(f"  node_order         : ordenacao alfabetica — train_mean/std alinhados com tensor")
log.info(
    f"  config.py          : freq_min={CFG.freq_min} | "
    f"in_steps=8 | out_steps=2 | num_layers=3 | "
    f"in_features={in_features_gerado}"
)
log.info(f"Pipeline v7 concluido — janela {CFG.freq_min} min.")