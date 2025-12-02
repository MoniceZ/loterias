from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline


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
    """Retorna médias históricas de desempenho dos modelos (ou pesos padrão se não existir histórico)."""
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
    for col in ["random_forest", "logistic_regression", "k_nearest_neighbors", "gradient_boosting"]:
        if col in df.columns:
            medias[col] = float(df[col].mean())

    # garante chaves padrão caso alguma coluna ainda não exista
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
    return (
        tipo_loteria.lower()
        .replace(" ", "_")
        .replace("+", "mais")
    )


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
def _preparar_ml(df: pd.DataFrame, tipo_loteria: str) -> Tuple[np.ndarray, np.ndarray, int]:
    cols = [c for c in df.columns if c.startswith("num_")]
    arr = df[cols].astype(int)
    max_d = _max_num_por_tipo(tipo_loteria, df)

    X: List[List[int]] = []
    Y: List[List[int]] = []

    if len(arr) < 2:
        return np.asarray(X), np.asarray(Y), max_d

    for i in range(len(arr) - 1):
        atual = arr.iloc[i].tolist()
        prox = arr.iloc[i + 1].tolist()

        linha = [0] * (max_d + 1)
        for d in atual:
            if 0 <= d <= max_d:
                linha[int(d)] = 1

        prox_limpo = [int(x) for x in prox if 0 <= int(x) <= max_d]
        Y.append(sorted(set(prox_limpo)))
        X.append(linha)

    if not X or not Y:
        return np.asarray(X), np.asarray(Y), max_d

    mlb = MultiLabelBinarizer(classes=list(range(max_d + 1)))
    Y_bin = mlb.fit_transform(Y)
    return np.asarray(X), Y_bin, max_d


def _avaliar_modelo(clf, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    if len(X) < 3:
        return 0.0, 0.0

    # para estratificação, usamos a classe mais provável (argmax) como rótulo
    y_estrato = Y.argmax(axis=1)
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    f1s: List[float] = []
    accs: List[float] = []

    for train_idx, test_idx in kf.split(X, y_estrato):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        f1s.append(f1_score(y_test, pred, average="micro"))
        accs.append(accuracy_score(y_test, pred))

    return float(np.mean(f1s)), float(np.mean(accs))


def _rank_por_modelo(
    df: pd.DataFrame,
    top_n: int,
    base_estimator,
    tipo_loteria: str,
    nome_modelo: str,
) -> Tuple[List[int], float]:
    X, Y, max_d = _preparar_ml(df, tipo_loteria)

    # poucos dados ou sem labels válidos -> usa heurística em vez de sequência 1..N
    if len(X) < 10 or Y.size == 0:
        # aqui devolvemos a predição de frequência para não cair em 1,2,3,4,5,6...
        return predizer_por_frequencia(df, top_n), 0.0

    # carrega modelo salvo (se existir) para comparação
    salvo = _tentar_carregar_modelo(tipo_loteria, nome_modelo)
    pipeline_antigo = salvo.get("pipeline") if isinstance(salvo, dict) else None
    n_features_salvo = salvo.get("n_features") if isinstance(salvo, dict) else None

    melhor_pipeline = None
    melhor_score = 0.0

    # avalia modelo antigo com todos os dados atuais
    if pipeline_antigo is not None and n_features_salvo == X.shape[1]:
        try:
            f1_old, acc_old = _avaliar_modelo(pipeline_antigo, X, Y)
            melhor_score = (f1_old + acc_old) / 2.0
            melhor_pipeline = pipeline_antigo
        except Exception:
            melhor_score = 0.0
            melhor_pipeline = None

    # cria novo pipeline
    pipeline_novo = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(0.95, X.shape[1]))),
            ("clf", OneVsRestClassifier(base_estimator)),
        ]
    )

    f1_new, acc_new = _avaliar_modelo(pipeline_novo, X, Y)
    score_novo = (f1_new + acc_new) / 2.0

    if score_novo >= melhor_score:
        # novo modelo é melhor (ou não existia modelo antigo)
        melhor_score = score_novo
        melhor_pipeline = pipeline_novo

    # re-treina o melhor pipeline com todos os dados disponíveis
    melhor_pipeline.fit(X, Y)

    # salva pipeline atualizado no disco para uso futuro
    _salvar_modelo(
        tipo_loteria,
        nome_modelo,
        {
            "pipeline": melhor_pipeline,
            "n_features": int(X.shape[1]),
            "max_num": int(max_d),
            "n_samples": int(len(X)),
        },
    )

    # ranking final de dezenas com base na probabilidade para a última linha de X
    try:
        scores = melhor_pipeline.predict_proba(X)[-1]
    except Exception:
        try:
            scores = melhor_pipeline.decision_function(X)[-1]
        except Exception:
            scores = np.random.rand(max_d + 1)

    scores = np.asarray(scores)
    if scores.shape[0] > max_d + 1:
        scores = scores[: max_d + 1]

    # descartamos a posição 0 (não existe dezena 0)
    scores_sem_zero = scores[1:]
    idx = np.argsort(scores_sem_zero)[-top_n:][::-1]
    dezenas = [int(i + 1) for i in idx]  # volta para 1..max_d

    dezenas = sorted(dezenas)[:top_n]

    return dezenas, float(melhor_score)


# ---------- modelos ----------
def predizer_random_forest(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "random_forest")


def predizer_logistic(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=42,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "logistic_regression")


def predizer_knn(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    clf = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "k_nearest_neighbors")


def predizer_gb(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "gradient_boosting")


# ---------- orquestra ----------
def gerar_palpite(df: pd.DataFrame, tipo_loteria: str) -> Dict[str, List[int]]:
    top_n = _top_n_por_tipo(tipo_loteria)

    # histórico médio para ajustar pesos
    historico = _carregar_historico()

    # heurísticas
    freq = predizer_por_frequencia(df, top_n)
    rec = predizer_por_recencia(df, top_n)

    # modelos com persistência + auto-avaliação
    rf, rf_acc = predizer_random_forest(df, top_n, tipo_loteria)
    lg, lg_acc = predizer_logistic(df, top_n, tipo_loteria)
    kn, kn_acc = predizer_knn(df, top_n, tipo_loteria)
    gb, gb_acc = predizer_gb(df, top_n, tipo_loteria)

    resultados: Dict[str, List[int]] = {
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

    # pesos combinam histórico e desempenho atual (autoajuste)
    pesos_modelos: Dict[str, float] = {
        "frequencia_simples": 0.5,
        "recencia_ponderada": 0.8,
        "random_forest": 1.0 + (rf_acc + historico.get("random_forest", 1.0)) / 2.0,
        "logistic_regression": 0.9 + (lg_acc + historico.get("logistic_regression", 1.0)) / 2.0,
        "k_nearest_neighbors": 0.8 + (kn_acc + historico.get("k_nearest_neighbors", 1.0)) / 2.0,
        "gradient_boosting": 1.1 + (gb_acc + historico.get("gradient_boosting", 1.0)) / 2.0,
    }

    votos: Counter[int] = Counter()
    for nome, lista in resultados.items():
        peso = pesos_modelos.get(nome, 1.0)
        for d in lista:
            votos[int(d)] += peso

    melhores = sorted([d for d, _ in votos.most_common(top_n)])

    resultados["melhor_combinacao"] = melhores
    resultados["avaliacao_modelos"] = {k: float(round(v, 4)) for k, v in desempenho.items()}

    # salva histórico atualizado (para reforço em execuções futuras)
    _salvar_historico(tipo_loteria, desempenho)

    return resultados
