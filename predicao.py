from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score


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


def carregar_dados(caminho_csv: str | Path) -> pd.DataFrame:
    p = Path(caminho_csv)
    df = pd.read_csv(p)
    dezenas_cols = [c for c in df.columns if c.startswith("num_")]
    df[dezenas_cols] = df[dezenas_cols].astype(int)
    return df


def _salvar_historico(tipo_loteria: str, desempenho: Dict[str, float], caminho: str = "historico_modelos.csv"):
    """
    Salva ou atualiza o histórico de desempenho em CSV.
    """
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
    Retorna médias históricas de desempenho dos modelos (ou pesos padrão se não existir histórico).
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
    medias = {}
    for col in ["random_forest", "logistic_regression", "k_nearest_neighbors", "gradient_boosting"]:
        if col in df.columns:
            medias[col] = float(df[col].mean())
    return medias


# ---------- heurísticas ----------
def predizer_por_frequencia(df: pd.DataFrame, top_n: int) -> List[int]:
    dezenas = df[[c for c in df.columns if c.startswith("num_")]].to_numpy().ravel()
    cont = Counter(map(int, dezenas))
    return sorted([d for d, _ in cont.most_common(top_n)])


def predizer_por_recencia(df: pd.DataFrame, top_n: int) -> List[int]:
    cols = [c for c in df.columns if c.startswith("num_")]
    pesos = np.geomspace(1.0, 0.01, num=len(df))
    cont: Counter[int] = Counter()
    for w, row in zip(pesos, df[cols].to_numpy()):
        for d in row:
            cont[int(d)] += float(w)
    return sorted([d for d, _ in cont.most_common(top_n)])


# ---------- ML base ----------
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


def _avaliar_modelo(clf, X, Y) -> Tuple[float, float]:
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1s, accs = [], []

    for train_idx, test_idx in kf.split(X, Y.argmax(axis=1)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        f1s.append(f1_score(y_test, pred, average="micro"))
        accs.append(accuracy_score(y_test, pred))

    return float(np.mean(f1s)), float(np.mean(accs))


def _rank_por_modelo(df: pd.DataFrame, top_n: int, base_estimator) -> Tuple[List[int], float]:
    X, Y, max_d = _preparar_ml(df)
    if len(X) < 10:
        return list(range(1, min(top_n, max_d) + 1)), 0.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=min(0.95, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    clf = OneVsRestClassifier(base_estimator)

    f1, acc = _avaliar_modelo(clf, X_pca, Y)
    clf.fit(X_pca, Y)

    try:
        scores = clf.predict_proba([X_pca[-1]])[0]
    except Exception:
        try:
            scores = clf.decision_function([X_pca[-1]])[0]
        except Exception:
            scores = np.random.rand(max_d + 1)

    idx = np.argsort(scores)[-top_n:][::-1]
    return sorted(int(i) for i in idx), (f1 + acc) / 2


# ---------- modelos ----------
def predizer_random_forest(df: pd.DataFrame, top_n: int) -> Tuple[List[int], float]:
    return _rank_por_modelo(df, top_n, RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=42
    ))


def predizer_logistic(df: pd.DataFrame, top_n: int) -> Tuple[List[int], float]:
    return _rank_por_modelo(df, top_n, LogisticRegression(
        max_iter=3000, solver="lbfgs", random_state=42
    ))


def predizer_knn(df: pd.DataFrame, top_n: int) -> Tuple[List[int], float]:
    return _rank_por_modelo(df, top_n, KNeighborsClassifier(
        n_neighbors=5, weights="distance"
    ))


def predizer_gb(df: pd.DataFrame, top_n: int) -> Tuple[List[int], float]:
    return _rank_por_modelo(df, top_n, GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
    ))


# ---------- orquestra ----------
def gerar_palpite(df: pd.DataFrame, tipo_loteria: str) -> Dict[str, List[int]]:
    top_n = _top_n_por_tipo(tipo_loteria)

    # histórico médio para ajustar pesos
    historico = _carregar_historico()

    freq = predizer_por_frequencia(df, top_n)
    rec = predizer_por_recencia(df, top_n)
    rf, rf_acc = predizer_random_forest(df, top_n)
    lg, lg_acc = predizer_logistic(df, top_n)
    kn, kn_acc = predizer_knn(df, top_n)
    gb, gb_acc = predizer_gb(df, top_n)

    resultados = {
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

    # pesos combinam histórico e desempenho atual
    pesos_modelos = {
        "frequencia_simples": 0.5,
        "recencia_ponderada": 0.8,
        "random_forest": 1.0 + (rf_acc + historico["random_forest"]) / 2,
        "logistic_regression": 0.9 + (lg_acc + historico["logistic_regression"]) / 2,
        "k_nearest_neighbors": 0.8 + (kn_acc + historico["k_nearest_neighbors"]) / 2,
        "gradient_boosting": 1.1 + (gb_acc + historico["gradient_boosting"]) / 2,
    }

    votos: Counter[int] = Counter()
    for nome, lista in resultados.items():
        peso = pesos_modelos.get(nome, 1)
        for d in lista:
            votos[int(d)] += peso

    melhores = sorted([d for d, _ in votos.most_common(top_n)])

    resultados["melhor_combinacao"] = melhores
    resultados["avaliacao_modelos"] = {
        k: round(v, 4) for k, v in desempenho.items()
    }

    # salva histórico atualizado
    _salvar_historico(tipo_loteria, desempenho)

    return resultados
