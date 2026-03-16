from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =========================
# Tuning (sem mudar API)
# =========================
_RANDOM_STATE = 42

# Você pode “pesar a mão” sem editar o arquivo:
#   Windows (PowerShell):  $env:LOTO_RF_N_ESTIMATORS="2000"
#   Linux/macOS:          export LOTO_RF_N_ESTIMATORS=2000
_LOTO_LOOKBACK = int(np.clip(int(__import__("os").environ.get("LOTO_LOOKBACK", "3")), 1, 20))
_LOTO_CV_SPLITS = int(np.clip(int(__import__("os").environ.get("LOTO_CV_SPLITS", "5")), 3, 10))
_LOTO_RF_N_ESTIMATORS = int(np.clip(int(__import__("os").environ.get("LOTO_RF_N_ESTIMATORS", "1200")), 200, 5000))
_LOTO_GB_N_ESTIMATORS = int(np.clip(int(__import__("os").environ.get("LOTO_GB_N_ESTIMATORS", "800")), 100, 5000))

_MODELOS_AVALIADOS = (
    "frequencia_simples",
    "recencia_ponderada",
    "random_forest",
    "logistic_regression",
    "k_nearest_neighbors",
    "gradient_boosting",
)


# ---------- util ----------
def _is_super_sete(tipo_loteria: str) -> bool:
    t = (tipo_loteria or "").lower()
    return ("super" in t) and ("sete" in t)


def _top_n_por_tipo(tipo_loteria: str) -> int:
    t = tipo_loteria.lower()
    if "facil" in t:
        return 15
    if "mega" in t:
        return 6
    if "quina" in t:
        return 5
    if "lotomania" in t:
        return 50
    if _is_super_sete(tipo_loteria):
        return 7
    return 20


def _limites_dezenas_por_tipo(tipo_loteria: str) -> Tuple[int, int, int]:
    """
    Retorna (min_dezenas, max_dezenas, min_num).

    min_num:
      - 1 na maioria (dezenas 1..N)
      - 0 na Lotomania (00..99 -> 0..99)
      - 0 no Super Sete (0..9 por coluna)
    """
    t = tipo_loteria.lower()

    # Regras típicas de aposta (quantidade de dezenas escolhidas)
    if "mega" in t:
        return 6, 20, 1
    if "facil" in t:
        return 15, 20, 1
    if "quina" in t:
        return 5, 15, 1
    if "lotomania" in t:
        return 50, 50, 0
    if _is_super_sete(tipo_loteria):
        return 7, 7, 0

    # fallback genérico
    return 1, 60, 1


def _ajustar_n_dezenas(
    tipo_loteria: str,
    n_dezenas: int | None,
    max_num: int,
) -> Tuple[int, int, int, int, List[str]]:
    """
    Ajusta n_dezenas para respeitar os limites por tipo e o universo disponível.

    Retorna:
      (n_usada, min_dezenas, max_dezenas, min_num, avisos)
    """
    min_dezenas, max_dezenas, min_num = _limites_dezenas_por_tipo(tipo_loteria)
    avisos: List[str] = []

    if n_dezenas is None:
        n_usada = _top_n_por_tipo(tipo_loteria)
    else:
        n_usada = int(n_dezenas)

    universo_total = int(max_num - min_num + 1)
    if max_dezenas > universo_total:
        max_dezenas = universo_total

    original = n_usada

    if n_usada < min_dezenas:
        n_usada = min_dezenas
    if n_usada > max_dezenas:
        n_usada = max_dezenas

    if original != n_usada:
        avisos.append(
            f"n_dezenas ajustado de {original} para {n_usada} "
            f"(limites: {min_dezenas}..{max_dezenas})"
        )

    return n_usada, min_dezenas, max_dezenas, min_num, avisos


def _ordenar_dataframe_temporalmente(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "concurso" in df.columns:
        concurso = pd.to_numeric(df["concurso"], errors="coerce")
        if concurso.notna().any():
            df["concurso"] = concurso
            return df.sort_values("concurso").reset_index(drop=True)

    if "data" in df.columns:
        datas = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
        if datas.notna().any():
            return (
                df.assign(_ord_data=datas)
                .sort_values("_ord_data")
                .drop(columns="_ord_data")
                .reset_index(drop=True)
            )

    return df.reset_index(drop=True)


def carregar_dados(caminho_csv: str | Path) -> pd.DataFrame:
    p = Path(caminho_csv)
    df = pd.read_csv(p)
    dezenas_cols = [c for c in df.columns if c.startswith("num_")]
    df[dezenas_cols] = df[dezenas_cols].astype(int)
    return _ordenar_dataframe_temporalmente(df)


def _salvar_historico(
    tipo_loteria: str,
    desempenho: Dict[str, float],
    caminho: str = "historico_modelos.csv",
) -> None:
    """Salva ou atualiza o histórico de desempenho em CSV."""
    data = {
        "data_execucao": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo_loteria": tipo_loteria,
    }
    data.update(desempenho)

    historico_path = Path(caminho)
    df_novo = pd.DataFrame([data])

    if historico_path.exists():
        df_antigo = pd.read_csv(historico_path)
        df_final = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        df_final = df_novo

    df_final.to_csv(historico_path, index=False)


def _carregar_historico(caminho: str = "historico_modelos.csv") -> Dict[str, float]:
    """
    Retorna médias históricas de desempenho dos modelos (ou pesos padrão
    se não existir histórico).
    """
    p = Path(caminho)
    if not p.exists():
        return {
            "random_forest": 1.0,
            "logistic_regression": 1.0,
            "k_nearest_neighbors": 1.0,
            "gradient_boosting": 1.0,
        }

    df = pd.read_csv(p)
    medias: Dict[str, float] = {}
    for col in [
        "random_forest",
        "logistic_regression",
        "k_nearest_neighbors",
        "gradient_boosting",
    ]:
        if col in df.columns:
            medias[col] = float(df[col].mean())

    for nome, padrao in [
        ("random_forest", 1.0),
        ("logistic_regression", 1.0),
        ("k_nearest_neighbors", 1.0),
        ("gradient_boosting", 1.0),
    ]:
        medias.setdefault(nome, padrao)

    return medias


# ---------- persistência de modelos ----------
def _carregar_historico(
    tipo_loteria: str | None = None,
    caminho: str = "historico_modelos.csv",
) -> Dict[str, float]:
    """
    Retorna mÃ©dias histÃ³ricas de desempenho dos modelos.

    Quando `tipo_loteria` Ã© informado, filtra apenas o histÃ³rico daquela
    loteria para evitar misturar desempenhos de jogos diferentes.
    """
    medias = {nome: 0.0 for nome in _MODELOS_AVALIADOS}
    p = Path(caminho)

    if not p.exists():
        return medias

    df = pd.read_csv(p)

    if tipo_loteria and "tipo_loteria" in df.columns:
        df = df[df["tipo_loteria"].astype(str).str.lower() == tipo_loteria.lower()]

    if df.empty:
        return medias

    if "data_execucao" in df.columns:
        datas = pd.to_datetime(df["data_execucao"], errors="coerce")
        df = (
            df.assign(_ord_data=datas)
            .sort_values("_ord_data", na_position="last")
            .drop(columns="_ord_data")
            .tail(24)
            .reset_index(drop=True)
        )

    pesos = np.linspace(1.0, 2.5, num=len(df), dtype=np.float64)

    for col in _MODELOS_AVALIADOS:
        if col in df.columns:
            serie = pd.to_numeric(df[col], errors="coerce").dropna()
            if not serie.empty:
                pesos_validos = pesos[-len(serie) :]
                medias[col] = float(np.average(serie.to_numpy(dtype=np.float64), weights=pesos_validos))
            else:
                medias[col] = 0.0

    return medias


def _slug_tipo(tipo_loteria: str) -> str:
    return tipo_loteria.lower().replace(" ", "_").replace("+", "mais")


def _modelo_path(tipo_loteria: str, nome_modelo: str) -> Path:
    slug = _slug_tipo(tipo_loteria)
    return Path("modelos") / f"{slug}_{nome_modelo}.joblib"


def _salvar_modelo(tipo_loteria: str, nome_modelo: str, payload: dict) -> None:
    caminho = _modelo_path(tipo_loteria, nome_modelo)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    dump(payload, caminho)


def _tentar_carregar_modelo(tipo_loteria: str, nome_modelo: str) -> dict | None:
    caminho = _modelo_path(tipo_loteria, nome_modelo)
    if not caminho.exists():
        return None
    try:
        return load(caminho)
    except Exception:
        return None


def _max_num_por_tipo(tipo_loteria: str, df: pd.DataFrame) -> int:
    t = tipo_loteria.lower()
    if "facil" in t:
        return 25
    if "mega" in t:
        return 60
    if "quina" in t:
        return 80
    if "lotomania" in t:
        return 99
    if _is_super_sete(tipo_loteria):
        return 9

    cols = [c for c in df.columns if c.startswith("num_")]
    arr = df[cols].astype(int).to_numpy()
    return int(arr.max())


# ---------- heurísticas (vetorizadas e com recência correta) ----------
def predizer_por_frequencia(df: pd.DataFrame, top_n: int) -> List[int]:
    cols = [c for c in df.columns if c.startswith("num_")]
    dezenas = df[cols].to_numpy(dtype=np.int64).ravel()
    dezenas = dezenas[dezenas >= 0]
    if dezenas.size == 0:
        return []
    cont = np.bincount(dezenas, minlength=int(dezenas.max()) + 1).astype(np.float64)
    nums = np.arange(cont.size, dtype=np.int64)
    order = np.lexsort((nums, -cont))  # (cont desc, num asc)
    escolhidos = nums[order][: int(top_n)]
    return sorted(int(x) for x in escolhidos.tolist())


def predizer_por_recencia(df: pd.DataFrame, top_n: int) -> List[int]:
    cols = [c for c in df.columns if c.startswith("num_")]
    arr = df[cols].to_numpy(dtype=np.int64)
    if arr.size == 0:
        return []
    n, k = arr.shape

    # Mais recente => maior peso (assumindo df em ordem cronológica crescente)
    pesos = np.geomspace(0.01, 1.0, num=n).astype(np.float64)

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    vals = arr.ravel()
    mask = vals >= 0
    vals = vals[mask]
    w = pesos[rows[mask]]

    cont = np.bincount(vals, weights=w, minlength=int(vals.max()) + 1).astype(np.float64)
    nums = np.arange(cont.size, dtype=np.int64)
    order = np.lexsort((nums, -cont))
    escolhidos = nums[order][: int(top_n)]
    return sorted(int(x) for x in escolhidos.tolist())


# ---------- Super Sete (heurísticas por POSIÇÃO) ----------
def _cols_num(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("num_")])


def predizer_supersete_por_frequencia(df: pd.DataFrame) -> List[int]:
    cols = _cols_num(df)
    if not cols:
        return []
    arr = df[cols].to_numpy(dtype=np.int64)
    if arr.size == 0:
        return []
    palp: List[int] = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        col = col[(col >= 0) & (col <= 9)]
        if col.size == 0:
            palp.append(0)
            continue
        cont = np.bincount(col, minlength=10).astype(np.float64)
        m = cont.max()
        candidatos = np.flatnonzero(cont == m)
        palp.append(int(candidatos.min()) if candidatos.size else 0)
    return palp


def predizer_supersete_por_recencia(df: pd.DataFrame) -> List[int]:
    cols = _cols_num(df)
    if not cols:
        return []
    arr = df[cols].to_numpy(dtype=np.int64)
    if arr.size == 0:
        return []
    n = arr.shape[0]
    pesos = np.geomspace(0.01, 1.0, num=n).astype(np.float64)

    palp: List[int] = []
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask = (col >= 0) & (col <= 9)
        colv = col[mask]
        w = pesos[mask]
        if colv.size == 0:
            palp.append(0)
            continue
        cont = np.bincount(colv, weights=w, minlength=10).astype(np.float64)
        m = cont.max()
        candidatos = np.flatnonzero(cont == m)
        palp.append(int(candidatos.min()) if candidatos.size else 0)
    return palp


# ---------- ML base (mais eficiente + prevê o PRÓXIMO sorteio de verdade) ----------
def _binarizar_sorteios(arr: np.ndarray, max_d: int) -> np.ndarray:
    """
    Converte matriz (n_sorteios, k_dezenas) em matriz binária (n_sorteios, max_d+1).
    """
    arr = np.asarray(arr, dtype=np.int64)
    n, k = arr.shape
    B = np.zeros((n, max_d + 1), dtype=np.uint8)

    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    vals = arr.ravel()
    mask = (vals >= 0) & (vals <= max_d)
    B[rows[mask], vals[mask]] = 1
    return B


def _n_features_temporais(base_dim: int) -> int:
    return int(base_dim * 7)


def _janela_normalizada(cumsum_pad: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, window))
    starts = np.maximum(0, ends - (window - 1))
    counts = (cumsum_pad[ends + 1] - cumsum_pad[starts]).astype(np.float32)
    lens = (ends - starts + 1).astype(np.float32).reshape(-1, 1)
    return counts / np.maximum(lens, 1.0)


def _calcular_atrasos_binarios(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=np.uint8)
    if B.size == 0:
        return np.empty_like(B, dtype=np.float32)

    n, m = B.shape
    cap = max(10, min(60, n))
    atrasos = np.empty((n, m), dtype=np.float32)
    ultimo = np.full(m, cap, dtype=np.float32)

    for i in range(n):
        ultimo = np.minimum(ultimo + 1.0, float(cap))
        ativos = B[i].astype(bool)
        ultimo[ativos] = 0.0
        atrasos[i] = ultimo

    return atrasos / float(cap)


def _montar_features_temporais(B: np.ndarray) -> np.ndarray:
    B = np.asarray(B, dtype=np.uint8)
    if B.size == 0:
        return np.empty((0, 0), dtype=np.float32)

    Bf = B.astype(np.float32)
    n = B.shape[0]
    L = int(max(1, _LOTO_LOOKBACK))
    cumsum = np.cumsum(Bf, axis=0, dtype=np.float32)
    cumsum_pad = np.vstack([np.zeros((1, B.shape[1]), dtype=np.float32), cumsum])
    ends = np.arange(n, dtype=np.int64)

    freq_recente = _janela_normalizada(cumsum_pad, ends, L)
    freq_media = _janela_normalizada(cumsum_pad, ends, max(5, L * 2))
    freq_longa = _janela_normalizada(cumsum_pad, ends, max(10, L * 4))
    freq_global = cumsum / np.arange(1, n + 1, dtype=np.float32).reshape(-1, 1)
    atrasos = _calcular_atrasos_binarios(B)
    tendencia = freq_recente - freq_global

    return np.hstack(
        [Bf, freq_recente, freq_media, freq_longa, freq_global, atrasos, tendencia]
    ).astype(np.float32)


def _indices_walk_forward(n_samples: int) -> List[int]:
    if n_samples < 8:
        return []

    min_train = max(6, min(24, n_samples // 2))
    if min_train >= n_samples:
        return []

    candidatos = np.arange(min_train, n_samples, dtype=np.int64)
    max_splits = min(_LOTO_CV_SPLITS, len(candidatos))
    if max_splits <= 0:
        return []

    if len(candidatos) <= max_splits:
        return [int(i) for i in candidatos.tolist()]

    selecionados = np.linspace(
        int(candidatos[0]),
        int(candidatos[-1]),
        num=max_splits,
        dtype=np.int64,
    )
    return [int(i) for i in np.unique(selecionados).tolist()]


def _dezenas_lista_para_binario(
    dezenas: List[int],
    max_d: int,
    tipo_loteria: str,
) -> np.ndarray:
    _, _, min_num = _limites_dezenas_por_tipo(tipo_loteria)
    out = np.zeros((max_d + 1,), dtype=np.uint8)
    for dezena in dezenas:
        d = int(dezena)
        if min_num <= d <= max_d:
            out[d] = 1
    return out


def _metricas_vazias(supersete: bool = False) -> Dict[str, float]:
    if supersete:
        return {
            "score_composto": 0.0,
            "taxa_acerto": 0.0,
            "f1_ponderado": 0.0,
            "acerto_4_mais": 0.0,
            "acerto_exato": 0.0,
            "acertos_medios": 0.0,
            "amostras_avaliadas": 0.0,
        }

    return {
        "score_composto": 0.0,
        "taxa_acerto": 0.0,
        "precisao_aposta": 0.0,
        "jaccard": 0.0,
        "f1_micro": 0.0,
        "acuracia_exata": 0.0,
        "acertos_medios": 0.0,
        "amostras_avaliadas": 0.0,
    }


def _consolidar_metricas_multilabel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return _metricas_vazias()

    y_true = np.asarray(y_true, dtype=np.uint8)
    y_pred = np.asarray(y_pred, dtype=np.uint8)

    acertos = (y_true & y_pred).sum(axis=1).astype(np.float32)
    reais = np.maximum(1.0, y_true.sum(axis=1).astype(np.float32))
    apostadas = np.maximum(1.0, y_pred.sum(axis=1).astype(np.float32))
    uniao = np.maximum(1.0, (y_true | y_pred).sum(axis=1).astype(np.float32))

    taxa_acerto = float(np.mean(acertos / reais))
    precisao_aposta = float(np.mean(acertos / apostadas))
    jaccard = float(np.mean(acertos / uniao))
    f1_micro = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    acuracia_exata = float(accuracy_score(y_true, y_pred))
    acertos_medios = float(np.mean(acertos))

    score_composto = float(
        np.clip(
            0.34 * taxa_acerto
            + 0.22 * precisao_aposta
            + 0.22 * jaccard
            + 0.17 * f1_micro
            + 0.05 * acuracia_exata,
            0.0,
            1.0,
        )
    )

    return {
        "score_composto": score_composto,
        "taxa_acerto": taxa_acerto,
        "precisao_aposta": precisao_aposta,
        "jaccard": jaccard,
        "f1_micro": f1_micro,
        "acuracia_exata": acuracia_exata,
        "acertos_medios": acertos_medios,
        "amostras_avaliadas": float(y_true.shape[0]),
    }


def _consolidar_metricas_supersete(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return _metricas_vazias(supersete=True)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acertos = (y_true == y_pred).sum(axis=1).astype(np.float32)
    taxa_acerto = float(np.mean(acertos / 7.0))
    f1_ponderado = float(
        f1_score(y_true.ravel(), y_pred.ravel(), average="weighted", zero_division=0)
    )
    acerto_4_mais = float(np.mean(acertos >= 4))
    acerto_exato = float(np.mean(np.all(y_true == y_pred, axis=1)))
    acertos_medios = float(np.mean(acertos))

    score_composto = float(
        np.clip(
            0.45 * taxa_acerto
            + 0.30 * f1_ponderado
            + 0.20 * acerto_4_mais
            + 0.05 * acerto_exato,
            0.0,
            1.0,
        )
    )

    return {
        "score_composto": score_composto,
        "taxa_acerto": taxa_acerto,
        "f1_ponderado": f1_ponderado,
        "acerto_4_mais": acerto_4_mais,
        "acerto_exato": acerto_exato,
        "acertos_medios": acertos_medios,
        "amostras_avaliadas": float(y_true.shape[0]),
    }


def _preparar_ml(
    df: pd.DataFrame, tipo_loteria: str
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    cols = [c for c in df.columns if c.startswith("num_")]
    arr = df[cols].astype(int).to_numpy(dtype=np.int64)
    max_d = _max_num_por_tipo(tipo_loteria, df)
    n_features = _n_features_temporais(max_d + 1)

    if arr.shape[0] < 2:
        vazioX = np.empty((0, n_features), dtype=np.float32)
        vazioY = np.empty((0, max_d + 1), dtype=np.uint8)
        return vazioX, vazioY, max_d, np.zeros((n_features,), dtype=np.float32)

    B = _binarizar_sorteios(arr, max_d=max_d)  # (n_sorteios, max_d+1)

    # Labels: próximo sorteio (i -> i+1)
    X_full = _montar_features_temporais(B)
    Y = B[1:].astype(np.uint8)

    # Features: janela lookback até o sorteio atual
    L = int(max(1, _LOTO_LOOKBACK))
    cumsum = np.cumsum(B, axis=0, dtype=np.int16)
    cumsum_pad = np.vstack([np.zeros((1, max_d + 1), dtype=np.int16), cumsum])

    # X para pares (i -> i+1), i = 0..n-2
    n_pairs = B.shape[0] - 1
    ends = np.arange(n_pairs, dtype=np.int64)  # 0..n-2
    starts = np.maximum(0, ends - (L - 1))
    X = (cumsum_pad[ends + 1] - cumsum_pad[starts]).astype(np.float32)

    # x_pred: usa o último sorteio real (e lookback) para prever o próximo
    end_last = B.shape[0] - 1
    start_last = max(0, end_last - (L - 1))
    x_pred = (cumsum_pad[end_last + 1] - cumsum_pad[start_last]).astype(np.float32)

    return X, Y, max_d, x_pred


# ---------- ML Super Sete (por posição, 7 saídas 0..9) ----------
def _preparar_ml_supersete(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    cols = _cols_num(df)
    arr = df[cols].astype(int).to_numpy(dtype=np.int64)
    max_d = 9

    if arr.shape[0] < 2:
        vazioX = np.empty((0, 70), dtype=np.float32)
        vazioY = np.empty((0, 7), dtype=np.int64)
        return vazioX, vazioY, max_d, np.zeros((70,), dtype=np.float32)

    # One-hot por posição: (n, 7*10)
    n = arr.shape[0]
    B = np.zeros((n, 70), dtype=np.uint8)
    for i in range(n):
        for pos in range(min(7, arr.shape[1])):
            v = int(arr[i, pos])
            if 0 <= v <= 9:
                B[i, pos * 10 + v] = 1

    # Labels: próximo sorteio (i -> i+1) como inteiros por posição
    Y = arr[1:, :7].astype(np.int64)

    # Features: janela lookback (soma de one-hots)
    L = int(max(1, _LOTO_LOOKBACK))
    cumsum = np.cumsum(B, axis=0, dtype=np.int16)
    cumsum_pad = np.vstack([np.zeros((1, 70), dtype=np.int16), cumsum])

    n_pairs = B.shape[0] - 1
    ends = np.arange(n_pairs, dtype=np.int64)  # 0..n-2
    starts = np.maximum(0, ends - (L - 1))
    X = (cumsum_pad[ends + 1] - cumsum_pad[starts]).astype(np.float32)

    end_last = B.shape[0] - 1
    start_last = max(0, end_last - (L - 1))
    x_pred = (cumsum_pad[end_last + 1] - cumsum_pad[start_last]).astype(np.float32)

    return X, Y, max_d, x_pred


def _estrato_proxy(Y: np.ndarray) -> np.ndarray:
    """
    StratifiedKFold precisa de uma classe 1D com contagens mínimas por classe.
    Para multilabel, usamos proxy estável: bins por quantis do maior número do target.
    """
    y_max = Y.argmax(axis=1).astype(np.int64)
    if y_max.size < 30:
        return y_max

    qs = np.quantile(y_max, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    bins = np.unique(qs)
    if bins.size < 3:
        return y_max
    return np.digitize(y_max, bins[1:-1], right=True).astype(np.int64)


def _avaliar_modelo(clf, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    if len(X) < 10:
        return 0.0, 0.0

    y_estrato = _estrato_proxy(Y)

    # n_splits seguro
    counts = np.bincount(y_estrato) if y_estrato.size else np.array([0], dtype=np.int64)
    min_count = int(counts.min()) if counts.size else 0
    n_splits = min(_LOTO_CV_SPLITS, 5, min_count) if min_count >= 3 else 3
    n_splits = max(3, min(n_splits, len(X)))

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=_RANDOM_STATE)

    f1s: List[float] = []
    accs: List[float] = []

    for train_idx, test_idx in kf.split(X, y_estrato):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        f1s.append(f1_score(y_test, pred, average="micro", zero_division=0))
        accs.append(accuracy_score(y_test, pred))

    return float(np.mean(f1s)), float(np.mean(accs))


def _avaliar_modelo_supersete(clf, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    # Y shape: (n, 7), classes: 0..9
    if len(X) < 10 or Y.size == 0:
        return 0.0, 0.0

    y_proxy = Y[:, 0].astype(np.int64)  # estratifica pelo 1º dígito (0..9)
    counts = np.bincount(y_proxy, minlength=10)
    counts_nonzero = counts[counts > 0]
    if counts_nonzero.size == 0:
        return 0.0, 0.0
    min_count = int(counts_nonzero.min())

    n_splits = min(_LOTO_CV_SPLITS, 5, min_count, len(X))
    n_splits = max(2, n_splits)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=_RANDOM_STATE)

    f1s: List[float] = []
    accs: List[float] = []

    for train_idx, test_idx in kf.split(X, y_proxy):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        f1s.append(f1_score(y_test.ravel(), pred.ravel(), average="micro", zero_division=0))
        accs.append(float((pred == y_test).mean()))

    return float(np.mean(f1s)), float(np.mean(accs))


def _preparar_ml(
    df: pd.DataFrame, tipo_loteria: str
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    cols = [c for c in df.columns if c.startswith("num_")]
    arr = df[cols].astype(int).to_numpy(dtype=np.int64)
    max_d = _max_num_por_tipo(tipo_loteria, df)
    n_features = _n_features_temporais(max_d + 1)

    if arr.shape[0] < 2:
        vazioX = np.empty((0, n_features), dtype=np.float32)
        vazioY = np.empty((0, max_d + 1), dtype=np.uint8)
        return vazioX, vazioY, max_d, np.zeros((n_features,), dtype=np.float32)

    B = _binarizar_sorteios(arr, max_d=max_d)
    X_full = _montar_features_temporais(B)
    Y = B[1:].astype(np.uint8)
    X = X_full[:-1].astype(np.float32)
    x_pred = X_full[-1].astype(np.float32)
    return X, Y, max_d, x_pred


def _preparar_ml_supersete(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    cols = _cols_num(df)
    arr = df[cols].astype(int).to_numpy(dtype=np.int64)
    max_d = 9
    n_features = _n_features_temporais(70)

    if arr.shape[0] < 2:
        vazioX = np.empty((0, n_features), dtype=np.float32)
        vazioY = np.empty((0, 7), dtype=np.int64)
        return vazioX, vazioY, max_d, np.zeros((n_features,), dtype=np.float32)

    n = arr.shape[0]
    B = np.zeros((n, 70), dtype=np.uint8)
    for i in range(n):
        for pos in range(min(7, arr.shape[1])):
            v = int(arr[i, pos])
            if 0 <= v <= 9:
                B[i, pos * 10 + v] = 1

    X_full = _montar_features_temporais(B)
    Y = arr[1:, :7].astype(np.int64)
    X = X_full[:-1].astype(np.float32)
    x_pred = X_full[-1].astype(np.float32)
    return X, Y, max_d, x_pred


def _avaliar_modelo(
    clf,
    X: np.ndarray,
    Y: np.ndarray,
    top_n: int,
    tipo_loteria: str,
    max_d: int,
) -> Dict[str, float]:
    if len(X) < 8 or Y.size == 0:
        return _metricas_vazias()

    y_true_amostras: List[np.ndarray] = []
    y_pred_amostras: List[np.ndarray] = []

    for idx in _indices_walk_forward(len(X)):
        X_train = X[:idx]
        y_train = Y[:idx]
        if len(X_train) == 0:
            continue

        clf.fit(X_train, y_train)
        x_test = np.asarray(X[idx], dtype=np.float32).reshape(1, -1)
        scores = _extrair_scores(clf, x_test, max_d=max_d)
        dezenas = _selecionar_top_dezenas_por_scores(scores, top_n, tipo_loteria, max_d)

        y_true_amostras.append(np.asarray(Y[idx], dtype=np.uint8))
        y_pred_amostras.append(_dezenas_lista_para_binario(dezenas, max_d, tipo_loteria))

    if not y_true_amostras:
        return _metricas_vazias()

    return _consolidar_metricas_multilabel(
        np.vstack(y_true_amostras),
        np.vstack(y_pred_amostras),
    )


def _avaliar_modelo_supersete(
    clf,
    X: np.ndarray,
    Y: np.ndarray,
) -> Dict[str, float]:
    if len(X) < 8 or Y.size == 0:
        return _metricas_vazias(supersete=True)

    y_true_amostras: List[np.ndarray] = []
    y_pred_amostras: List[np.ndarray] = []

    for idx in _indices_walk_forward(len(X)):
        X_train = X[:idx]
        y_train = Y[:idx]
        if len(X_train) == 0:
            continue

        clf.fit(X_train, y_train)
        x_test = np.asarray(X[idx], dtype=np.float32).reshape(1, -1)
        pred = np.asarray(clf.predict(x_test), dtype=np.int64).reshape(1, -1)

        y_true_amostras.append(np.asarray(Y[idx], dtype=np.int64))
        y_pred_amostras.append(pred[0, :7].astype(np.int64))

    if not y_true_amostras:
        return _metricas_vazias(supersete=True)

    return _consolidar_metricas_supersete(
        np.vstack(y_true_amostras),
        np.vstack(y_pred_amostras),
    )


def _selecionar_top_dezenas_por_scores(
    scores: np.ndarray,
    top_n: int,
    tipo_loteria: str,
    max_d: int,
) -> List[int]:
    """
    Seleciona as top_n dezenas conforme scores.

    - Para Lotomania: permite 0 (00).
    - Para demais: ignora posição 0 (não existe dezena 0).
    """
    _, _, min_num = _limites_dezenas_por_tipo(tipo_loteria)

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.shape[0] > max_d + 1:
        scores = scores[: max_d + 1]
    elif scores.shape[0] < max_d + 1:
        pad_val = float(scores.min()) if scores.size else 0.0
        scores = np.pad(scores, (0, (max_d + 1) - scores.shape[0]), constant_values=pad_val)

    if min_num == 0:
        idx = np.argsort(scores)[-top_n:][::-1]
        dezenas = [int(i) for i in idx]
        return sorted(dezenas)[:top_n]

    scores_sem_zero = scores[1:]
    idx = np.argsort(scores_sem_zero)[-top_n:][::-1]
    dezenas = [int(i + 1) for i in idx]
    return sorted(dezenas)[:top_n]


def _extrair_scores(model, x_pred: np.ndarray, max_d: int) -> np.ndarray:
    """
    Unifica saída de:
      - OneVsRest (array shape [n_classes])
      - Multioutput (lista de arrays por saída)
    """
    try:
        proba = model.predict_proba(x_pred)
        if isinstance(proba, list):
            # multioutput => lista; cada item: (n_samples, n_classes)
            out = np.empty((len(proba),), dtype=np.float64)
            for i, p in enumerate(proba):
                if p.ndim == 2 and p.shape[1] >= 2:
                    out[i] = float(p[0, 1])
                else:
                    out[i] = float(p[0, 0]) if p.size else 0.0
            return out
        return np.asarray(proba[0], dtype=np.float64)
    except Exception:
        try:
            df = model.decision_function(x_pred)
            if isinstance(df, list):
                out = np.asarray([float(d[0]) for d in df], dtype=np.float64)
                return out
            return np.asarray(df[0], dtype=np.float64)
        except Exception:
            return np.random.default_rng(_RANDOM_STATE).random(max_d + 1)


def _extrair_scores_supersete(model, x_pred: np.ndarray) -> List[np.ndarray]:
    """
    Retorna lista de 7 vetores de score (len=10) para cada posição.
    """
    # Descobrir o estimador final (pipeline ou estimator direto)
    clf = model
    if hasattr(model, "named_steps"):
        clf = model.named_steps.get("clf", model)

    try:
        proba = model.predict_proba(x_pred)
    except Exception:
        # fallback: aleatório estável
        rng = np.random.default_rng(_RANDOM_STATE)
        return [rng.random(10) for _ in range(7)]

    if not isinstance(proba, list):
        # inesperado; tenta coerção
        proba_list = [np.asarray(proba, dtype=np.float64)]
    else:
        proba_list = proba

    # classes por saída, quando disponível
    classes_list = None
    if hasattr(clf, "classes_"):
        classes_list = clf.classes_
    elif hasattr(clf, "estimators_"):
        try:
            classes_list = [est.classes_ for est in clf.estimators_]
        except Exception:
            classes_list = None

    scores_out: List[np.ndarray] = []
    for i in range(7):
        # proba[i]: (n_samples, n_classes_i)
        if i >= len(proba_list):
            scores_out.append(np.zeros(10, dtype=np.float64))
            continue

        p = np.asarray(proba_list[i], dtype=np.float64)
        if p.ndim == 2:
            p = p[0]
        p = p.reshape(-1)

        scores = np.zeros(10, dtype=np.float64)
        if classes_list is not None and i < len(classes_list):
            try:
                cls = np.asarray(classes_list[i], dtype=np.int64).reshape(-1)
                for c, s in zip(cls.tolist(), p.tolist()):
                    if 0 <= int(c) <= 9:
                        scores[int(c)] = float(s)
            except Exception:
                # fallback simples: mapeia na ordem 0..len(p)-1
                for c in range(min(10, p.size)):
                    scores[c] = float(p[c])
        else:
            for c in range(min(10, p.size)):
                scores[c] = float(p[c])

        scores_out.append(scores)

    # garante 7 posições
    while len(scores_out) < 7:
        scores_out.append(np.zeros(10, dtype=np.float64))

    return scores_out


def _selecionar_digitos_supersete(scores_list: List[np.ndarray]) -> List[int]:
    palp: List[int] = []
    for sc in scores_list[:7]:
        sc = np.asarray(sc, dtype=np.float64).reshape(-1)
        if sc.size < 10:
            sc = np.pad(sc, (0, 10 - sc.size), constant_values=float(sc.min()) if sc.size else 0.0)
        m = float(sc.max()) if sc.size else 0.0
        cand = np.flatnonzero(sc == m)
        palp.append(int(cand.min()) if cand.size else 0)
    return palp


def _rank_por_modelo(
    df: pd.DataFrame,
    top_n: int,
    base_estimator,
    tipo_loteria: str,
    nome_modelo: str,
    usar_ovr: bool,
) -> Tuple[List[int], float]:
    X, Y, max_d, x_pred_vec = _preparar_ml(df, tipo_loteria)

    if len(X) < 10 or Y.size == 0:
        return predizer_por_frequencia(df, top_n), 0.0

    salvo = _tentar_carregar_modelo(tipo_loteria, nome_modelo)
    pipeline_antigo = salvo.get("pipeline") if isinstance(salvo, dict) else None
    n_features_salvo = salvo.get("n_features") if isinstance(salvo, dict) else None

    melhor_pipeline = None
    melhor_score = 0.0

    if pipeline_antigo is not None and n_features_salvo == X.shape[1]:
        try:
            f1_old, acc_old = _avaliar_modelo(pipeline_antigo, X, Y)
            melhor_score = (f1_old + acc_old) / 2.0
            melhor_pipeline = pipeline_antigo
        except Exception:
            melhor_score = 0.0
            melhor_pipeline = None

    usar_pca = X.shape[1] >= 10 and X.shape[0] >= 30
    pca_step = PCA(n_components=0.95) if usar_pca else "passthrough"

    clf_step = OneVsRestClassifier(base_estimator, n_jobs=-1) if usar_ovr else base_estimator

    pipeline_novo = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", pca_step),
            ("clf", clf_step),
        ]
    )

    f1_new, acc_new = _avaliar_modelo(pipeline_novo, X, Y)
    score_novo = (f1_new + acc_new) / 2.0

    if score_novo >= melhor_score:
        melhor_score = score_novo
        melhor_pipeline = pipeline_novo

    melhor_pipeline.fit(X, Y)

    _salvar_modelo(
        tipo_loteria,
        nome_modelo,
        {
            "pipeline": melhor_pipeline,
            "n_features": int(X.shape[1]),
            "max_num": int(max_d),
            "n_samples": int(len(X)),
            "lookback": int(_LOTO_LOOKBACK),
            "ovr": bool(usar_ovr),
        },
    )

    # Previsão do PRÓXIMO sorteio: usa o último sorteio real como entrada
    x_pred = np.asarray(x_pred_vec, dtype=np.float32).reshape(1, -1)
    scores = _extrair_scores(melhor_pipeline, x_pred, max_d=max_d)

    dezenas = _selecionar_top_dezenas_por_scores(scores, top_n, tipo_loteria, max_d)
    return dezenas, float(melhor_score)


def _rank_por_modelo_supersete(
    df: pd.DataFrame,
    base_estimator,
    tipo_loteria: str,
    nome_modelo: str,
) -> Tuple[List[int], float]:
    X, Y, max_d, x_pred_vec = _preparar_ml_supersete(df)

    if len(X) < 10 or Y.size == 0:
        return predizer_supersete_por_frequencia(df), 0.0

    salvo = _tentar_carregar_modelo(tipo_loteria, nome_modelo)
    pipeline_antigo = salvo.get("pipeline") if isinstance(salvo, dict) else None
    n_features_salvo = salvo.get("n_features") if isinstance(salvo, dict) else None

    melhor_pipeline = None
    melhor_score = 0.0

    if pipeline_antigo is not None and n_features_salvo == X.shape[1]:
        try:
            f1_old, acc_old = _avaliar_modelo_supersete(pipeline_antigo, X, Y)
            melhor_score = (f1_old + acc_old) / 2.0
            melhor_pipeline = pipeline_antigo
        except Exception:
            melhor_score = 0.0
            melhor_pipeline = None

    usar_pca = X.shape[1] >= 10 and X.shape[0] >= 30
    pca_step = PCA(n_components=0.95) if usar_pca else "passthrough"

    pipeline_novo = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", pca_step),
            ("clf", base_estimator),
        ]
    )

    f1_new, acc_new = _avaliar_modelo_supersete(pipeline_novo, X, Y)
    score_novo = (f1_new + acc_new) / 2.0

    if score_novo >= melhor_score:
        melhor_score = score_novo
        melhor_pipeline = pipeline_novo

    melhor_pipeline.fit(X, Y)

    _salvar_modelo(
        tipo_loteria,
        nome_modelo,
        {
            "pipeline": melhor_pipeline,
            "n_features": int(X.shape[1]),
            "max_num": int(max_d),
            "n_samples": int(len(X)),
            "lookback": int(_LOTO_LOOKBACK),
            "ovr": False,
            "supersete": True,
        },
    )

    x_pred = np.asarray(x_pred_vec, dtype=np.float32).reshape(1, -1)
    scores_list = _extrair_scores_supersete(melhor_pipeline, x_pred)
    palp = _selecionar_digitos_supersete(scores_list)

    return palp, float(melhor_score)


def _avaliar_heuristica_multilabel(
    df: pd.DataFrame,
    tipo_loteria: str,
    top_n: int,
    pred_fn: Callable[[pd.DataFrame, int], List[int]],
) -> Dict[str, float]:
    cols = [c for c in df.columns if c.startswith("num_")]
    arr = df[cols].astype(int).to_numpy(dtype=np.int64)
    max_d = _max_num_por_tipo(tipo_loteria, df)

    if arr.shape[0] < 8:
        return _metricas_vazias()

    B = _binarizar_sorteios(arr, max_d=max_d)
    y_true_amostras: List[np.ndarray] = []
    y_pred_amostras: List[np.ndarray] = []

    for idx in _indices_walk_forward(len(df)):
        df_treino = df.iloc[:idx]
        if df_treino.empty:
            continue

        dezenas = pred_fn(df_treino, top_n)
        y_true_amostras.append(B[idx].astype(np.uint8))
        y_pred_amostras.append(_dezenas_lista_para_binario(dezenas, max_d, tipo_loteria))

    if not y_true_amostras:
        return _metricas_vazias()

    return _consolidar_metricas_multilabel(
        np.vstack(y_true_amostras),
        np.vstack(y_pred_amostras),
    )


def _avaliar_heuristica_supersete(
    df: pd.DataFrame,
    pred_fn: Callable[[pd.DataFrame], List[int]],
) -> Dict[str, float]:
    cols = _cols_num(df)
    arr = df[cols].astype(int).to_numpy(dtype=np.int64)

    if arr.shape[0] < 8:
        return _metricas_vazias(supersete=True)

    y_true_amostras: List[np.ndarray] = []
    y_pred_amostras: List[np.ndarray] = []

    for idx in _indices_walk_forward(len(df)):
        df_treino = df.iloc[:idx]
        if df_treino.empty:
            continue

        pred = np.asarray(pred_fn(df_treino), dtype=np.int64).reshape(-1)
        if pred.size < 7:
            pred = np.pad(pred, (0, 7 - pred.size), constant_values=0)

        y_true_amostras.append(arr[idx, :7].astype(np.int64))
        y_pred_amostras.append(pred[:7].astype(np.int64))

    if not y_true_amostras:
        return _metricas_vazias(supersete=True)

    return _consolidar_metricas_supersete(
        np.vstack(y_true_amostras),
        np.vstack(y_pred_amostras),
    )


def _peso_modelo_ensemble(score_atual: float, score_historico: float) -> float:
    score_atual = max(0.0, float(score_atual))
    score_historico = max(0.0, float(score_historico))
    return float(max(0.15, 0.25 + (0.85 * score_atual) + (0.45 * score_historico)))


def _resumo_modelos(
    avaliacao_modelos: Dict[str, float],
    pesos_modelos: Dict[str, float],
) -> Dict[str, Any]:
    if not avaliacao_modelos:
        return {
            "melhor_modelo": None,
            "score_melhor_modelo": 0.0,
            "confianca_relativa": 0.0,
            "peso_total_ensemble": 0.0,
        }

    ranking = sorted(
        ((str(nome), float(score)) for nome, score in avaliacao_modelos.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    melhor_modelo, melhor_score = ranking[0]
    segundo_score = ranking[1][1] if len(ranking) > 1 else 0.0
    gap = max(0.0, melhor_score - segundo_score)
    peso_total = float(sum(float(v) for v in pesos_modelos.values()))

    return {
        "melhor_modelo": melhor_modelo,
        "score_melhor_modelo": float(round(melhor_score, 4)),
        "confianca_relativa": float(round(gap, 4)),
        "peso_total_ensemble": float(round(peso_total, 4)),
    }


def _gerar_alertas_dados_fracos(
    tipo_loteria: str,
    total_concursos: int,
    avaliacao_detalhada: Dict[str, Dict[str, float]],
    resumo_modelos: Dict[str, Any],
    historico_modelos: Dict[str, float],
) -> List[Dict[str, str]]:
    alertas: List[Dict[str, str]] = []

    min_concursos_recomendado = 60 if "lotofacil" in tipo_loteria.lower() else 40
    if total_concursos < min_concursos_recomendado:
        alertas.append(
            {
                "nivel": "warning",
                "titulo": "Base curta",
                "mensagem": (
                    f"Foram encontrados apenas {total_concursos} concursos. "
                    f"O ideal para essa analise e pelo menos {min_concursos_recomendado}."
                ),
            }
        )

    amostras_validas = [
        float(dados.get("amostras_avaliadas", 0.0))
        for dados in avaliacao_detalhada.values()
        if isinstance(dados, dict)
    ]
    if amostras_validas and min(amostras_validas) < 6:
        alertas.append(
            {
                "nivel": "warning",
                "titulo": "Pouca validacao temporal",
                "mensagem": (
                    "Alguns modelos foram avaliados com poucas amostras walk-forward. "
                    "Os scores podem oscilar bastante."
                ),
            }
        )

    confianca = float(resumo_modelos.get("confianca_relativa", 0.0) or 0.0)
    if confianca < 0.02:
        alertas.append(
            {
                "nivel": "info",
                "titulo": "Baixa separacao entre modelos",
                "mensagem": (
                    "Os modelos ficaram muito proximos em desempenho. "
                    "Considere a previsao como equilibrada e com menor confianca relativa."
                ),
            }
        )

    melhor_score = float(resumo_modelos.get("score_melhor_modelo", 0.0) or 0.0)
    if melhor_score < 0.08:
        alertas.append(
            {
                "nivel": "warning",
                "titulo": "Score baixo",
                "mensagem": (
                    "Mesmo o melhor modelo ficou com score baixo na validacao recente. "
                    "Vale interpretar os jogos sugeridos com cautela."
                ),
            }
        )

    if not any(float(v) > 0 for v in historico_modelos.values()):
        alertas.append(
            {
                "nivel": "info",
                "titulo": "Sem historico consolidado",
                "mensagem": (
                    "Ainda nao existe historico suficiente dessa loteria para calibrar bem os pesos do ensemble."
                ),
            }
        )

    return alertas


def _rank_por_modelo(
    df: pd.DataFrame,
    top_n: int,
    base_estimator,
    tipo_loteria: str,
    nome_modelo: str,
    usar_ovr: bool,
) -> Tuple[List[int], Dict[str, float]]:
    X, Y, max_d, x_pred_vec = _preparar_ml(df, tipo_loteria)

    if len(X) < 8 or Y.size == 0:
        return predizer_por_frequencia(df, top_n), _metricas_vazias()

    salvo = _tentar_carregar_modelo(tipo_loteria, nome_modelo)
    pipeline_antigo = salvo.get("pipeline") if isinstance(salvo, dict) else None
    n_features_salvo = salvo.get("n_features") if isinstance(salvo, dict) else None

    melhor_pipeline = None
    melhor_metricas = _metricas_vazias()

    if pipeline_antigo is not None and n_features_salvo == X.shape[1]:
        try:
            metricas_antigas = _avaliar_modelo(
                pipeline_antigo,
                X,
                Y,
                top_n=top_n,
                tipo_loteria=tipo_loteria,
                max_d=max_d,
            )
            melhor_metricas = metricas_antigas
            melhor_pipeline = pipeline_antigo
        except Exception:
            melhor_metricas = _metricas_vazias()
            melhor_pipeline = None

    usar_pca = X.shape[1] >= 10 and X.shape[0] >= 30
    pca_step = PCA(n_components=0.95) if usar_pca else "passthrough"
    clf_step = OneVsRestClassifier(base_estimator, n_jobs=-1) if usar_ovr else base_estimator

    pipeline_novo = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", pca_step),
            ("clf", clf_step),
        ]
    )

    metricas_novas = _avaliar_modelo(
        pipeline_novo,
        X,
        Y,
        top_n=top_n,
        tipo_loteria=tipo_loteria,
        max_d=max_d,
    )

    if metricas_novas.get("score_composto", 0.0) >= melhor_metricas.get("score_composto", 0.0):
        melhor_metricas = metricas_novas
        melhor_pipeline = pipeline_novo

    melhor_pipeline.fit(X, Y)

    _salvar_modelo(
        tipo_loteria,
        nome_modelo,
        {
            "pipeline": melhor_pipeline,
            "n_features": int(X.shape[1]),
            "max_num": int(max_d),
            "n_samples": int(len(X)),
            "lookback": int(_LOTO_LOOKBACK),
            "ovr": bool(usar_ovr),
        },
    )

    x_pred = np.asarray(x_pred_vec, dtype=np.float32).reshape(1, -1)
    scores = _extrair_scores(melhor_pipeline, x_pred, max_d=max_d)
    dezenas = _selecionar_top_dezenas_por_scores(scores, top_n, tipo_loteria, max_d)
    return dezenas, melhor_metricas


def _rank_por_modelo_supersete(
    df: pd.DataFrame,
    base_estimator,
    tipo_loteria: str,
    nome_modelo: str,
) -> Tuple[List[int], Dict[str, float]]:
    X, Y, max_d, x_pred_vec = _preparar_ml_supersete(df)

    if len(X) < 8 or Y.size == 0:
        return predizer_supersete_por_frequencia(df), _metricas_vazias(supersete=True)

    salvo = _tentar_carregar_modelo(tipo_loteria, nome_modelo)
    pipeline_antigo = salvo.get("pipeline") if isinstance(salvo, dict) else None
    n_features_salvo = salvo.get("n_features") if isinstance(salvo, dict) else None

    melhor_pipeline = None
    melhor_metricas = _metricas_vazias(supersete=True)

    if pipeline_antigo is not None and n_features_salvo == X.shape[1]:
        try:
            metricas_antigas = _avaliar_modelo_supersete(pipeline_antigo, X, Y)
            melhor_metricas = metricas_antigas
            melhor_pipeline = pipeline_antigo
        except Exception:
            melhor_metricas = _metricas_vazias(supersete=True)
            melhor_pipeline = None

    usar_pca = X.shape[1] >= 10 and X.shape[0] >= 30
    pca_step = PCA(n_components=0.95) if usar_pca else "passthrough"

    pipeline_novo = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", pca_step),
            ("clf", base_estimator),
        ]
    )

    metricas_novas = _avaliar_modelo_supersete(pipeline_novo, X, Y)

    if metricas_novas.get("score_composto", 0.0) >= melhor_metricas.get("score_composto", 0.0):
        melhor_metricas = metricas_novas
        melhor_pipeline = pipeline_novo

    melhor_pipeline.fit(X, Y)

    _salvar_modelo(
        tipo_loteria,
        nome_modelo,
        {
            "pipeline": melhor_pipeline,
            "n_features": int(X.shape[1]),
            "max_num": int(max_d),
            "n_samples": int(len(X)),
            "lookback": int(_LOTO_LOOKBACK),
            "ovr": False,
            "supersete": True,
        },
    )

    x_pred = np.asarray(x_pred_vec, dtype=np.float32).reshape(1, -1)
    scores_list = _extrair_scores_supersete(melhor_pipeline, x_pred)
    palp = _selecionar_digitos_supersete(scores_list)
    return palp, melhor_metricas


# ---------- modelos ----------
def predizer_random_forest(
    df: pd.DataFrame, top_n: int, tipo_loteria: str
) -> Tuple[List[int], Dict[str, float]]:
    if _is_super_sete(tipo_loteria):
        clf = RandomForestClassifier(
            n_estimators=_LOTO_RF_N_ESTIMATORS,
            max_depth=None,
            n_jobs=-1,
            random_state=_RANDOM_STATE,
            bootstrap=True,
        )
        return _rank_por_modelo_supersete(df, clf, tipo_loteria, "random_forest")

    # Multioutput nativo (bem mais eficiente que OneVsRest para dezenas/classes)
    clf = RandomForestClassifier(
        n_estimators=_LOTO_RF_N_ESTIMATORS,
        max_depth=None,
        n_jobs=-1,
        random_state=_RANDOM_STATE,
        bootstrap=True,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "random_forest", usar_ovr=False)


def predizer_logistic(
    df: pd.DataFrame, top_n: int, tipo_loteria: str
) -> Tuple[List[int], Dict[str, float]]:
    if _is_super_sete(tipo_loteria):
        base = LogisticRegression(
            max_iter=12000,
            solver="saga",
            penalty="l2",
            n_jobs=-1,
            random_state=_RANDOM_STATE,
        )
        clf = MultiOutputClassifier(base, n_jobs=-1)
        return _rank_por_modelo_supersete(df, clf, tipo_loteria, "logistic_regression")

    # OneVsRest necessário; solver saga usa múltiplos cores (n_jobs)
    clf = LogisticRegression(
        max_iter=12000,
        solver="saga",
        penalty="l2",
        n_jobs=-1,
        random_state=_RANDOM_STATE,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "logistic_regression", usar_ovr=True)


def predizer_knn(
    df: pd.DataFrame, top_n: int, tipo_loteria: str
) -> Tuple[List[int], Dict[str, float]]:
    if _is_super_sete(tipo_loteria):
        clf = KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1,
        )
        return _rank_por_modelo_supersete(df, clf, tipo_loteria, "k_nearest_neighbors")

    # Multioutput nativo
    clf = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        n_jobs=-1,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "k_nearest_neighbors", usar_ovr=False)


def predizer_gb(
    df: pd.DataFrame, top_n: int, tipo_loteria: str
) -> Tuple[List[int], Dict[str, float]]:
    if _is_super_sete(tipo_loteria):
        base = GradientBoostingClassifier(
            n_estimators=_LOTO_GB_N_ESTIMATORS,
            learning_rate=0.03,
            max_depth=5,
            random_state=_RANDOM_STATE,
        )
        clf = MultiOutputClassifier(base, n_jobs=-1)
        return _rank_por_modelo_supersete(df, clf, tipo_loteria, "gradient_boosting")

    # GradientBoosting não é multioutput => OneVsRest
    clf = GradientBoostingClassifier(
        n_estimators=_LOTO_GB_N_ESTIMATORS,
        learning_rate=0.03,
        max_depth=5,
        random_state=_RANDOM_STATE,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "gradient_boosting", usar_ovr=True)


# ---------- geração de jogos ----------
def _rank_dezenas_por_pontuacao(
    votos: Counter[int],
    min_num: int,
    max_num: int,
) -> List[int]:
    """
    Ordena dezenas min_num..max_num por pontuação (desc) e, em empate, por número (asc).
    """
    return sorted(
        range(min_num, max_num + 1),
        key=lambda d: (-float(votos.get(d, 0.0)), d),
    )


def _gerar_jogos_sugeridos(
    votos: Counter[int],
    min_num: int,
    max_num: int,
    n_jogos: int,
    n_dezenas: int,
) -> List[List[int]]:
    """
    Gera 'n_jogos' jogos, cada um com 'n_dezenas' dezenas, usando a pontuação do
    ensemble (votos) para ordenar e variar a parte "não-core".
    """
    if n_jogos < 1:
        n_jogos = 1
    if n_dezenas < 1:
        raise ValueError("n_dezenas deve ser >= 1.")

    universo_total = int(max_num - min_num + 1)
    if n_dezenas > universo_total:
        raise ValueError(
            f"n_dezenas ({n_dezenas}) não pode ser > universo ({universo_total})."
        )

    ranking = _rank_dezenas_por_pontuacao(votos, min_num, max_num)

    variar_qtd = max(1, n_dezenas // 3)
    core_size = max(0, n_dezenas - variar_qtd)

    if core_size >= len(ranking):
        core_size = max(0, len(ranking) - 1)

    core = ranking[:core_size]
    pool = ranking[core_size:]

    jogos: List[List[int]] = []
    vistos: set[Tuple[int, ...]] = set()

    for i in range(n_jogos):
        if not pool:
            jogo = tuple(sorted(core))
            jogos.append(list(jogo))
            continue

        start = (i * variar_qtd) % len(pool)
        escolhidas = list(core)

        j = 0
        while len(escolhidas) < n_dezenas and j < len(pool) * 2:
            d = pool[(start + j) % len(pool)]
            if d not in escolhidas:
                escolhidas.append(d)
            j += 1

        jogo_t = tuple(sorted(escolhidas))

        if jogo_t in vistos and len(pool) > 1:
            tries = 0
            alt_start = start
            while jogo_t in vistos and tries < min(10, len(pool)):
                alt_start = (alt_start + 1) % len(pool)
                escolhidas = list(core)
                j = 0
                while len(escolhidas) < n_dezenas and j < len(pool) * 2:
                    d = pool[(alt_start + j) % len(pool)]
                    if d not in escolhidas:
                        escolhidas.append(d)
                    j += 1
                jogo_t = tuple(sorted(escolhidas))
                tries += 1

        vistos.add(jogo_t)
        jogos.append(list(jogo_t))

    return jogos


def _gerar_jogos_sugeridos_supersete(
    votos_pos: List[Counter[int]],
    n_jogos: int,
) -> List[List[int]]:
    if n_jogos < 1:
        n_jogos = 1

    rankings: List[List[int]] = []
    for pos in range(7):
        c = votos_pos[pos] if pos < len(votos_pos) else Counter()
        ranking = sorted(range(0, 10), key=lambda d: (-float(c.get(d, 0.0)), d))
        rankings.append(ranking)

    jogos: List[List[int]] = []
    for i in range(n_jogos):
        jogo: List[int] = []
        for pos in range(7):
            rank = rankings[pos]
            idx = (i + pos) % len(rank)
            jogo.append(int(rank[idx]))
        jogos.append(jogo)
    return jogos


# ---------- orquestra ----------
def gerar_palpite(
    df: pd.DataFrame,
    tipo_loteria: str,
    n_jogos: int = 1,
    n_dezenas: int | None = None,
) -> Dict[str, Any]:
    """
    Mantém o comportamento atual (top_n por tipo) e adiciona:
      - n_jogos: quantos jogos gerar
      - n_dezenas: quantas dezenas por jogo (se None, usa o padrão por tipo)

    Correções:
      - Ajusta automaticamente n_dezenas para respeitar o mínimo/máximo por loteria
        (ex.: Lotofácil mínimo 15).
      - Lotomania permite 0 (00) nas dezenas.

    Super Sete:
      - Trata por posição (7 dígitos 0..9), permitindo repetição.
    """
    # --- Super Sete: fluxo próprio (por posição) ---
    if _is_super_sete(tipo_loteria):
        max_num = 9
        top_n, min_dez, max_dez, min_num, avisos = _ajustar_n_dezenas(
            tipo_loteria=tipo_loteria,
            n_dezenas=n_dezenas,
            max_num=max_num,
        )

        historico = _carregar_historico()

        freq = predizer_supersete_por_frequencia(df)
        rec = predizer_supersete_por_recencia(df)

        rf, rf_acc = predizer_random_forest(df, top_n, tipo_loteria)
        lg, lg_acc = predizer_logistic(df, top_n, tipo_loteria)
        kn, kn_acc = predizer_knn(df, top_n, tipo_loteria)
        gb, gb_acc = predizer_gb(df, top_n, tipo_loteria)

        resultados: Dict[str, Any] = {
            "frequencia_simples": freq,
            "recencia_ponderada": rec,
            "random_forest": rf,
            "logistic_regression": lg,
            "k_nearest_neighbors": kn,
            "gradient_boosting": gb,
        }

        desempenho = {
            "random_forest": rf_acc,
            "logistic_regression": lg_acc,
            "k_nearest_neighbors": kn_acc,
            "gradient_boosting": gb_acc,
        }

        pesos_modelos: Dict[str, float] = {
            "frequencia_simples": 0.5,
            "recencia_ponderada": 0.8,
            "random_forest": 1.0 + (rf_acc + historico.get("random_forest", 1.0)) / 2.0,
            "logistic_regression": (
                0.9 + (lg_acc + historico.get("logistic_regression", 1.0)) / 2.0
            ),
            "k_nearest_neighbors": (
                0.8 + (kn_acc + historico.get("k_nearest_neighbors", 1.0)) / 2.0
            ),
            "gradient_boosting": (
                1.1 + (gb_acc + historico.get("gradient_boosting", 1.0)) / 2.0
            ),
        }

        votos_pos: List[Counter[int]] = [Counter() for _ in range(7)]
        for nome, lista in resultados.items():
            peso = float(pesos_modelos.get(nome, 1.0))
            if not isinstance(lista, list):
                continue
            for pos in range(min(7, len(lista))):
                votos_pos[pos][int(lista[pos])] += peso

        melhores: List[int] = []
        for pos in range(7):
            c = votos_pos[pos]
            if not c:
                melhores.append(0)
                continue
            ranking = sorted(range(0, 10), key=lambda d: (-float(c.get(d, 0.0)), d))
            melhores.append(int(ranking[0]))

        resultados["melhor_combinacao"] = melhores
        resultados["avaliacao_modelos"] = {k: float(round(v, 4)) for k, v in desempenho.items()}

        resultados["jogos_sugeridos"] = _gerar_jogos_sugeridos_supersete(
            votos_pos=votos_pos,
            n_jogos=max(1, int(n_jogos)),
        )

        resultados["parametros"] = {
            "tipo_loteria": tipo_loteria,
            "n_jogos": int(max(1, int(n_jogos))),
            "n_dezenas_solicitada": None if n_dezenas is None else int(n_dezenas),
            "n_dezenas_usada": int(top_n),
            "min_dezenas": int(min_dez),
            "max_dezenas": int(max_dez),
            "min_num": int(min_num),
            "max_num": int(max_num),
            "lookback_usado": int(_LOTO_LOOKBACK),
            "cv_splits_max": int(_LOTO_CV_SPLITS),
            "rf_n_estimators": int(_LOTO_RF_N_ESTIMATORS),
            "gb_n_estimators": int(_LOTO_GB_N_ESTIMATORS),
        }
        if avisos:
            resultados["avisos"] = avisos

        _salvar_historico(tipo_loteria, desempenho)
        return resultados

    # --- Fluxo original (demais loterias) ---
    max_num = _max_num_por_tipo(tipo_loteria, df)
    top_n, min_dez, max_dez, min_num, avisos = _ajustar_n_dezenas(
        tipo_loteria=tipo_loteria,
        n_dezenas=n_dezenas,
        max_num=max_num,
    )

    historico = _carregar_historico()

    freq = predizer_por_frequencia(df, top_n)
    rec = predizer_por_recencia(df, top_n)

    rf, rf_acc = predizer_random_forest(df, top_n, tipo_loteria)
    lg, lg_acc = predizer_logistic(df, top_n, tipo_loteria)
    kn, kn_acc = predizer_knn(df, top_n, tipo_loteria)
    gb, gb_acc = predizer_gb(df, top_n, tipo_loteria)

    resultados: Dict[str, Any] = {
        "frequencia_simples": freq,
        "recencia_ponderada": rec,
        "random_forest": rf,
        "logistic_regression": lg,
        "k_nearest_neighbors": kn,
        "gradient_boosting": gb,
    }

    desempenho = {
        "random_forest": rf_acc,
        "logistic_regression": lg_acc,
        "k_nearest_neighbors": kn_acc,
        "gradient_boosting": gb_acc,
    }

    pesos_modelos: Dict[str, float] = {
        "frequencia_simples": 0.5,
        "recencia_ponderada": 0.8,
        "random_forest": 1.0 + (rf_acc + historico.get("random_forest", 1.0)) / 2.0,
        "logistic_regression": (
            0.9 + (lg_acc + historico.get("logistic_regression", 1.0)) / 2.0
        ),
        "k_nearest_neighbors": (
            0.8 + (kn_acc + historico.get("k_nearest_neighbors", 1.0)) / 2.0
        ),
        "gradient_boosting": (
            1.1 + (gb_acc + historico.get("gradient_boosting", 1.0)) / 2.0
        ),
    }

    votos: Counter[int] = Counter()
    for nome, lista in resultados.items():
        peso = float(pesos_modelos.get(nome, 1.0))
        for d in lista:
            votos[int(d)] += peso

    melhores = sorted([d for d, _ in votos.most_common(top_n)])

    resultados["melhor_combinacao"] = melhores
    resultados["avaliacao_modelos"] = {k: float(round(v, 4)) for k, v in desempenho.items()}

    resultados["jogos_sugeridos"] = _gerar_jogos_sugeridos(
        votos=votos,
        min_num=min_num,
        max_num=max_num,
        n_jogos=max(1, int(n_jogos)),
        n_dezenas=top_n,
    )

    resultados["parametros"] = {
        "tipo_loteria": tipo_loteria,
        "n_jogos": int(max(1, int(n_jogos))),
        "n_dezenas_solicitada": None if n_dezenas is None else int(n_dezenas),
        "n_dezenas_usada": int(top_n),
        "min_dezenas": int(min_dez),
        "max_dezenas": int(max_dez),
        "min_num": int(min_num),
        "max_num": int(max_num),
        "lookback_usado": int(_LOTO_LOOKBACK),
        "cv_splits_max": int(_LOTO_CV_SPLITS),
        "rf_n_estimators": int(_LOTO_RF_N_ESTIMATORS),
        "gb_n_estimators": int(_LOTO_GB_N_ESTIMATORS),
    }
    if avisos:
        resultados["avisos"] = avisos

    _salvar_historico(tipo_loteria, desempenho)

    return resultados


def gerar_palpite(
    df: pd.DataFrame,
    tipo_loteria: str,
    n_jogos: int = 1,
    n_dezenas: int | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Dict[str, Any]:
    df = _ordenar_dataframe_temporalmente(df)

    def _arredondar_metricas(metricas: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        return {
            nome: {chave: float(round(valor, 4)) for chave, valor in dados.items()}
            for nome, dados in metricas.items()
        }

    def _report(progress: int, mensagem: str = "") -> None:
        if progress_callback is None:
            return
        progress_callback(int(max(0, min(100, progress))), mensagem)

    historico = _carregar_historico(tipo_loteria)
    n_jogos_ajustado = max(1, int(n_jogos))
    _report(5, "Preparando dados e histórico de desempenho...")

    if _is_super_sete(tipo_loteria):
        max_num = 9
        top_n, min_dez, max_dez, min_num, avisos = _ajustar_n_dezenas(
            tipo_loteria=tipo_loteria,
            n_dezenas=n_dezenas,
            max_num=max_num,
        )

        _report(12, "Calculando heuristicas de frequencia e recencia...")
        freq = predizer_supersete_por_frequencia(df)
        rec = predizer_supersete_por_recencia(df)
        _report(28, "Executando Random Forest...")
        rf, rf_metricas = predizer_random_forest(df, top_n, tipo_loteria)
        _report(46, "Executando Regressao Logistica...")
        lg, lg_metricas = predizer_logistic(df, top_n, tipo_loteria)
        _report(62, "Executando KNN...")
        kn, kn_metricas = predizer_knn(df, top_n, tipo_loteria)
        _report(78, "Executando Gradient Boosting...")
        gb, gb_metricas = predizer_gb(df, top_n, tipo_loteria)

        resultados: Dict[str, Any] = {
            "frequencia_simples": freq,
            "recencia_ponderada": rec,
            "random_forest": rf,
            "logistic_regression": lg,
            "k_nearest_neighbors": kn,
            "gradient_boosting": gb,
        }

        avaliacao_detalhada: Dict[str, Dict[str, float]] = {
            "frequencia_simples": _avaliar_heuristica_supersete(df, predizer_supersete_por_frequencia),
            "recencia_ponderada": _avaliar_heuristica_supersete(df, predizer_supersete_por_recencia),
            "random_forest": rf_metricas,
            "logistic_regression": lg_metricas,
            "k_nearest_neighbors": kn_metricas,
            "gradient_boosting": gb_metricas,
        }

        avaliacao_modelos = {
            nome: float(round(metricas.get("score_composto", 0.0), 4))
            for nome, metricas in avaliacao_detalhada.items()
        }
        pesos_modelos = {
            nome: _peso_modelo_ensemble(
                score_atual=avaliacao_modelos.get(nome, 0.0),
                score_historico=historico.get(nome, 0.0),
            )
            for nome in resultados.keys()
        }

        votos_pos: List[Counter[int]] = [Counter() for _ in range(7)]
        for nome, lista in resultados.items():
            peso = float(pesos_modelos.get(nome, 1.0))
            if not isinstance(lista, list):
                continue
            for pos in range(min(7, len(lista))):
                votos_pos[pos][int(lista[pos])] += peso

        melhores: List[int] = []
        for pos in range(7):
            ranking = sorted(range(0, 10), key=lambda d: (-float(votos_pos[pos].get(d, 0.0)), d))
            melhores.append(int(ranking[0]) if ranking else 0)

        resultados["melhor_combinacao"] = melhores
        resultados["avaliacao_modelos"] = avaliacao_modelos
        resultados["avaliacao_detalhada"] = _arredondar_metricas(avaliacao_detalhada)
        resultados["historico_modelos"] = {
            nome: float(round(historico.get(nome, 0.0), 4)) for nome in avaliacao_modelos
        }
        resultados["pesos_modelos"] = {
            nome: float(round(pesos_modelos.get(nome, 0.0), 4)) for nome in pesos_modelos
        }
        resultados["resumo_modelos"] = _resumo_modelos(avaliacao_modelos, pesos_modelos)
        resultados["alertas_dados"] = _gerar_alertas_dados_fracos(
            tipo_loteria=tipo_loteria,
            total_concursos=int(len(df)),
            avaliacao_detalhada=avaliacao_detalhada,
            resumo_modelos=resultados["resumo_modelos"],
            historico_modelos=resultados["historico_modelos"],
        )
        resultados["avaliacao_metodo"] = (
            "walk-forward temporal + score composto por taxa de acerto, F1 e consistência"
        )
        _report(90, "Consolidando ensemble e sugerindo jogos...")
        resultados["jogos_sugeridos"] = _gerar_jogos_sugeridos_supersete(
            votos_pos=votos_pos,
            n_jogos=n_jogos_ajustado,
        )
        resultados["parametros"] = {
            "tipo_loteria": tipo_loteria,
            "n_jogos": int(n_jogos_ajustado),
            "n_dezenas_solicitada": None if n_dezenas is None else int(n_dezenas),
            "n_dezenas_usada": int(top_n),
            "min_dezenas": int(min_dez),
            "max_dezenas": int(max_dez),
            "min_num": int(min_num),
            "max_num": int(max_num),
            "lookback_usado": int(_LOTO_LOOKBACK),
            "cv_splits_max": int(_LOTO_CV_SPLITS),
            "rf_n_estimators": int(_LOTO_RF_N_ESTIMATORS),
            "gb_n_estimators": int(_LOTO_GB_N_ESTIMATORS),
        }
        if avisos:
            resultados["avisos"] = avisos

        _salvar_historico(tipo_loteria, avaliacao_modelos)
        _report(100, "Predicao concluida.")
        return resultados

    max_num = _max_num_por_tipo(tipo_loteria, df)
    top_n, min_dez, max_dez, min_num, avisos = _ajustar_n_dezenas(
        tipo_loteria=tipo_loteria,
        n_dezenas=n_dezenas,
        max_num=max_num,
    )

    _report(12, "Calculando heuristicas de frequencia e recencia...")
    freq = predizer_por_frequencia(df, top_n)
    rec = predizer_por_recencia(df, top_n)
    _report(28, "Executando Random Forest...")
    rf, rf_metricas = predizer_random_forest(df, top_n, tipo_loteria)
    _report(46, "Executando Regressao Logistica...")
    lg, lg_metricas = predizer_logistic(df, top_n, tipo_loteria)
    _report(62, "Executando KNN...")
    kn, kn_metricas = predizer_knn(df, top_n, tipo_loteria)
    _report(78, "Executando Gradient Boosting...")
    gb, gb_metricas = predizer_gb(df, top_n, tipo_loteria)

    resultados = {
        "frequencia_simples": freq,
        "recencia_ponderada": rec,
        "random_forest": rf,
        "logistic_regression": lg,
        "k_nearest_neighbors": kn,
        "gradient_boosting": gb,
    }

    avaliacao_detalhada = {
        "frequencia_simples": _avaliar_heuristica_multilabel(
            df,
            tipo_loteria,
            top_n,
            predizer_por_frequencia,
        ),
        "recencia_ponderada": _avaliar_heuristica_multilabel(
            df,
            tipo_loteria,
            top_n,
            predizer_por_recencia,
        ),
        "random_forest": rf_metricas,
        "logistic_regression": lg_metricas,
        "k_nearest_neighbors": kn_metricas,
        "gradient_boosting": gb_metricas,
    }

    avaliacao_modelos = {
        nome: float(round(metricas.get("score_composto", 0.0), 4))
        for nome, metricas in avaliacao_detalhada.items()
    }
    pesos_modelos = {
        nome: _peso_modelo_ensemble(
            score_atual=avaliacao_modelos.get(nome, 0.0),
            score_historico=historico.get(nome, 0.0),
        )
        for nome in resultados.keys()
    }

    votos: Counter[int] = Counter()
    for nome, lista in resultados.items():
        peso = float(pesos_modelos.get(nome, 1.0))
        for d in lista:
            votos[int(d)] += peso

    melhores = sorted(_rank_dezenas_por_pontuacao(votos, min_num, max_num)[:top_n])

    resultados["melhor_combinacao"] = melhores
    resultados["avaliacao_modelos"] = avaliacao_modelos
    resultados["avaliacao_detalhada"] = _arredondar_metricas(avaliacao_detalhada)
    resultados["historico_modelos"] = {
        nome: float(round(historico.get(nome, 0.0), 4)) for nome in avaliacao_modelos
    }
    resultados["pesos_modelos"] = {
        nome: float(round(pesos_modelos.get(nome, 0.0), 4)) for nome in pesos_modelos
    }
    resultados["resumo_modelos"] = _resumo_modelos(avaliacao_modelos, pesos_modelos)
    resultados["alertas_dados"] = _gerar_alertas_dados_fracos(
        tipo_loteria=tipo_loteria,
        total_concursos=int(len(df)),
        avaliacao_detalhada=avaliacao_detalhada,
        resumo_modelos=resultados["resumo_modelos"],
        historico_modelos=resultados["historico_modelos"],
    )
    resultados["avaliacao_metodo"] = (
        "walk-forward temporal + score composto por taxa de acerto, precisão da aposta e Jaccard"
    )
    _report(90, "Consolidando ensemble e sugerindo jogos...")
    resultados["jogos_sugeridos"] = _gerar_jogos_sugeridos(
        votos=votos,
        min_num=min_num,
        max_num=max_num,
        n_jogos=n_jogos_ajustado,
        n_dezenas=top_n,
    )
    resultados["parametros"] = {
        "tipo_loteria": tipo_loteria,
        "n_jogos": int(n_jogos_ajustado),
        "n_dezenas_solicitada": None if n_dezenas is None else int(n_dezenas),
        "n_dezenas_usada": int(top_n),
        "min_dezenas": int(min_dez),
        "max_dezenas": int(max_dez),
        "min_num": int(min_num),
        "max_num": int(max_num),
        "lookback_usado": int(_LOTO_LOOKBACK),
        "cv_splits_max": int(_LOTO_CV_SPLITS),
        "rf_n_estimators": int(_LOTO_RF_N_ESTIMATORS),
        "gb_n_estimators": int(_LOTO_GB_N_ESTIMATORS),
    }
    if avisos:
        resultados["avisos"] = avisos

    _salvar_historico(tipo_loteria, avaliacao_modelos)
    _report(100, "Predicao concluida.")
    return resultados


"""
USO (exemplo):

df = carregar_dados("mega_sena.csv")

# comportamento antigo (1 jogo, n_dezenas padrão do tipo)
res = gerar_palpite(df, "mega sena")

# agora escolhendo:
# - 5 jogos
# - 9 dezenas por jogo (se estiver fora dos limites do tipo, será ajustado)
res = gerar_palpite(df, "mega sena", n_jogos=5, n_dezenas=9)

print(res["jogos_sugeridos"])

# Para “forçar” mais processamento (sem editar o arquivo):
#   export LOTO_RF_N_ESTIMATORS=2500
#   export LOTO_GB_N_ESTIMATORS=1500
#   export LOTO_CV_SPLITS=7
#   export LOTO_LOOKBACK=5
"""
