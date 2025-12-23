from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
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


# ---------- util ----------
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
    return 20


def _limites_dezenas_por_tipo(tipo_loteria: str) -> Tuple[int, int, int]:
    """
    Retorna (min_dezenas, max_dezenas, min_num).

    min_num:
      - 1 na maioria (dezenas 1..N)
      - 0 na Lotomania (00..99 -> 0..99)
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


def carregar_dados(caminho_csv: str | Path) -> pd.DataFrame:
    p = Path(caminho_csv)
    df = pd.read_csv(p)
    dezenas_cols = [c for c in df.columns if c.startswith("num_")]
    df[dezenas_cols] = df[dezenas_cols].astype(int)
    return df


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


def _preparar_ml(
    df: pd.DataFrame, tipo_loteria: str
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    cols = [c for c in df.columns if c.startswith("num_")]
    arr = df[cols].astype(int).to_numpy(dtype=np.int64)
    max_d = _max_num_por_tipo(tipo_loteria, df)

    if arr.shape[0] < 2:
        vazio = np.empty((0, max_d + 1), dtype=np.float32)
        return vazio, vazio, max_d, np.zeros((max_d + 1,), dtype=np.float32)

    B = _binarizar_sorteios(arr, max_d=max_d)  # (n_sorteios, max_d+1)

    # Labels: próximo sorteio (i -> i+1)
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


# ---------- modelos ----------
def predizer_random_forest(
    df: pd.DataFrame, top_n: int, tipo_loteria: str
) -> Tuple[List[int], float]:
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
) -> Tuple[List[int], float]:
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
) -> Tuple[List[int], float]:
    # Multioutput nativo
    clf = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        n_jobs=-1,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "k_nearest_neighbors", usar_ovr=False)


def predizer_gb(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
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
    """
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
