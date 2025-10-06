from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer


# ---------- util ----------
def _top_n_por_tipo(tipo_loteria: str) -> int:
    t = tipo_loteria.lower()
    if "facil" in t:
        return 15
    if "mega" in t:
        return 6
    if "quina" in t:
        return 5
    return 20


def carregar_dados(caminho_csv: str | Path) -> pd.DataFrame:
    p = Path(caminho_csv)
    df = pd.read_csv(p)
    dezenas_cols = [c for c in df.columns if c.startswith("num_")]
    df[dezenas_cols] = df[dezenas_cols].astype(int)
    return df


# ---------- heurísticas ----------
def predizer_por_frequencia(df: pd.DataFrame, top_n: int) -> List[int]:
    dezenas = df[[c for c in df.columns if c.startswith("num_")]].to_numpy().ravel()
    cont = Counter(map(int, dezenas))
    return sorted([d for d, _ in cont.most_common(top_n)])


def predizer_por_recencia(df: pd.DataFrame, top_n: int) -> List[int]:
    cols = [c for c in df.columns if c.startswith("num_")]
    pesos = np.linspace(1.0, 0.1, num=len(df))
    cont: Counter[int] = Counter()
    for w, row in zip(pesos, df[cols].to_numpy()):
        for d in row:
            cont[int(d)] += float(w)
    return sorted([d for d, _ in cont.most_common(top_n)])


# ---------- ML ----------
def _preparar_ml(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int]:
    cols = [c for c in df.columns if c.startswith("num_")]
    arr = df[cols].astype(int)
    max_d = int(arr.to_numpy().max())
    X: List[List[int]] = []
    Y: List[List[int]] = []

    for i in range(len(arr) - 1):
        atual = arr.iloc[i].tolist()
        prox = arr.iloc[i + 1].tolist()
        linha = [0] * (max_d + 1)
        for d in atual:
            if 0 <= d <= max_d:
                linha[d] = 1
        X.append(linha)
        Y.append(sorted(set(int(x) for x in prox)))

    mlb = MultiLabelBinarizer(classes=list(range(max_d + 1)))
    Y_bin = mlb.fit_transform(Y)
    return np.asarray(X), Y_bin, max_d


def _rank_por_modelo(df: pd.DataFrame, top_n: int, base_estimator) -> List[int]:
    X, Y, max_d = _preparar_ml(df)
    if len(X) < 5:
        return list(range(1, min(top_n, max_d) + 1))

    X_train, X_test, y_train, _y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    clf = OneVsRestClassifier(base_estimator)
    clf.fit(X_train, y_train)

    scores = None
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba([X[-1]])[0]
    else:
        scores = clf.decision_function([X[-1]])[0]

    idx = np.argsort(scores)[-top_n:][::-1]
    return sorted(int(i) for i in idx)


def predizer_random_forest(df: pd.DataFrame, top_n: int) -> List[int]:
    return _rank_por_modelo(df, top_n, RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42))


def predizer_logistic(df: pd.DataFrame, top_n: int) -> List[int]:
    return _rank_por_modelo(df, top_n, LogisticRegression(max_iter=1000, n_jobs=None, random_state=42))


def predizer_knn(df: pd.DataFrame, top_n: int) -> List[int]:
    return _rank_por_modelo(df, top_n, KNeighborsClassifier(n_neighbors=3))


def predizer_gb(df: pd.DataFrame, top_n: int) -> List[int]:
    return _rank_por_modelo(df, top_n, GradientBoostingClassifier(n_estimators=100, random_state=42))


# ---------- orquestra ----------
def gerar_palpite(df: pd.DataFrame, tipo_loteria: str) -> Dict[str, List[int]]:
    top_n = _top_n_por_tipo(tipo_loteria)

    resultados = {
        "frequencia_simples": predizer_por_frequencia(df, top_n),
        "recencia_ponderada": predizer_por_recencia(df, top_n),
        "random_forest": predizer_random_forest(df, top_n),
        "logistic_regression": predizer_logistic(df, top_n),
        "k_nearest_neighbors": predizer_knn(df, top_n),
        "gradient_boosting": predizer_gb(df, top_n),
    }

    votos: Counter[int] = Counter()
    for lista in resultados.values():
        for d in lista:
            votos[int(d)] += 1

    resultados["melhor_combinacao"] = sorted([d for d, _ in votos.most_common(top_n)])
    return resultados
