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
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


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
    y_bin = mlb.fit_transform(Y)
    return np.asarray(X), y_bin, max_d


def _avaliar_modelo(clf, X: np.ndarray, Y: np.ndarray) -> Tuple[float, float]:
    if len(X) < 3:
        return 0.0, 0.0

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

    scores = np.asarray(scores).reshape(-1)
    if scores.shape[0] > max_d + 1:
        scores = scores[: max_d + 1]

    if min_num == 0:
        # 0..max_d
        idx = np.argsort(scores)[-top_n:][::-1]
        dezenas = [int(i) for i in idx]
        return sorted(dezenas)[:top_n]

    # 1..max_d
    scores_sem_zero = scores[1:]
    idx = np.argsort(scores_sem_zero)[-top_n:][::-1]
    dezenas = [int(i + 1) for i in idx]
    return sorted(dezenas)[:top_n]


def _rank_por_modelo(
    df: pd.DataFrame,
    top_n: int,
    base_estimator,
    tipo_loteria: str,
    nome_modelo: str,
) -> Tuple[List[int], float]:
    X, Y, max_d = _preparar_ml(df, tipo_loteria)

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
        },
    )

    try:
        scores = melhor_pipeline.predict_proba(X)[-1]
    except Exception:
        try:
            scores = melhor_pipeline.decision_function(X)[-1]
        except Exception:
            scores = np.random.rand(max_d + 1)

    dezenas = _selecionar_top_dezenas_por_scores(scores, top_n, tipo_loteria, max_d)
    return dezenas, float(melhor_score)


# ---------- modelos ----------
def predizer_random_forest(
    df: pd.DataFrame, top_n: int, tipo_loteria: str
) -> Tuple[List[int], float]:
    clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "random_forest")


def predizer_logistic(
    df: pd.DataFrame, top_n: int, tipo_loteria: str
) -> Tuple[List[int], float]:
    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        random_state=42,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "logistic_regression")


def predizer_knn(
    df: pd.DataFrame, top_n: int, tipo_loteria: str
) -> Tuple[List[int], float]:
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
"""
