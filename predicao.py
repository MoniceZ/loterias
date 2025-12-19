from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer, MaxAbsScaler
from sklearn.pipeline import Pipeline

# opcional (melhor para sparse)
try:
    from scipy import sparse
    _HAS_SCIPY = True
except Exception:
    sparse = None
    _HAS_SCIPY = False


# =========================
# CONFIG “PESADEIRO” (CPU/GPU)
# =========================
@dataclass(frozen=True)
class TrainConfig:
    lags: int = 8                 # mais lags => mais features => mais CPU
    ewm_alpha: float = 0.08       # recência exponencial (0..1) menor => mais “memória”
    cv_splits: int = 6            # backtesting pesado
    randomized_search_iter: int = 35  # tuning pesado (usa todos cores)
    use_random_search_min_samples: int = 180  # só faz tuning se tiver histórico suficiente
    random_state: int = 42


CFG = TrainConfig()


# =========================
# util
# =========================
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
    p = Path(caminho)
    # pesos base caso não haja histórico
    base = {
        "random_forest": 1.0,
        "extra_trees": 1.0,
        "logistic_regression": 1.0,
        "k_nearest_neighbors": 1.0,
        "hist_gradient_boosting": 1.0,
        "xgboost": 1.0,
    }

    if not p.exists():
        return base

    df = pd.read_csv(p)
    medias: Dict[str, float] = {}
    for col in base.keys():
        if col in df.columns:
            medias[col] = float(df[col].mean())
        else:
            medias[col] = base[col]
    return medias


# =========================
# persistência de modelos
# =========================
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


# =========================
# heurísticas
# =========================
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


# =========================
# FEATURES “PESADAS”
# =========================
def _onehot_row(nums: List[int], max_d: int) -> np.ndarray:
    v = np.zeros(max_d + 1, dtype=np.float32)
    for d in nums:
        if 0 <= int(d) <= max_d:
            v[int(d)] = 1.0
    return v


def _preparar_ml_pesado(
    df: pd.DataFrame,
    tipo_loteria: str,
    lags: int = CFG.lags,
    ewm_alpha: float = CFG.ewm_alpha,
) -> Tuple[Any, np.ndarray, int]:
    """
    X “bem maior”:
      - one-hot do concurso atual e de (lags-1) anteriores (concatenado)
      - frequência acumulada (rolling cumulativo) por dezena
      - recência exponencial (EWM) por dezena
    Y: dezenas do próximo concurso (multi-label binário)
    """
    cols = [c for c in df.columns if c.startswith("num_")]
    arr = df[cols].astype(int)
    max_d = _max_num_por_tipo(tipo_loteria, df)

    if len(arr) < 2:
        if _HAS_SCIPY:
            return sparse.csr_matrix((0, (max_d + 1) * lags + 2 * (max_d + 1))), np.zeros((0, max_d + 1), dtype=np.int8), max_d
        return np.zeros((0, (max_d + 1) * lags + 2 * (max_d + 1)), dtype=np.float32), np.zeros((0, max_d + 1), dtype=np.int8), max_d

    # transforma concursos em matriz one-hot (N x (max_d+1))
    oh = np.zeros((len(arr), max_d + 1), dtype=np.float32)
    for i in range(len(arr)):
        oh[i] = _onehot_row(arr.iloc[i].tolist(), max_d)

    # rolling cumulativo de frequência (normalizado)
    cumsum = np.cumsum(oh, axis=0)
    denom = np.arange(1, len(arr) + 1, dtype=np.float32).reshape(-1, 1)
    freq_cum = cumsum / denom

    # recência exponencial (EWM) aproximada via filtro recursivo
    ewm = np.zeros_like(oh, dtype=np.float32)
    prev = np.zeros((max_d + 1,), dtype=np.float32)
    a = float(ewm_alpha)
    for i in range(len(arr)):
        prev = a * oh[i] + (1.0 - a) * prev
        ewm[i] = prev

    # constrói X com lags
    X_blocks = []
    for k in range(lags):
        # shift para trás: X usa info até o concurso i, prediz i+1
        start = k
        end = len(arr) - 1  # último não tem próximo
        X_blocks.append(oh[start:end])

    # alinha blocos por tamanho mínimo (len(arr)-1-lags+1)
    min_len = len(arr) - 1 - (lags - 1)
    if min_len <= 0:
        # pouco histórico: volta para heurística depois
        if _HAS_SCIPY:
            return sparse.csr_matrix((0, 1)), np.zeros((0, 1), dtype=np.int8), max_d
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0, 1), dtype=np.int8), max_d

    X_lags = [b[:min_len] for b in X_blocks]  # cada um (min_len x (max_d+1))
    X_main = np.concatenate(X_lags, axis=1)   # (min_len x ((max_d+1)*lags))

    # adiciona features globais alinhadas ao “t” do último lag (i = (lags-1) ..)
    idx0 = lags - 1
    freq_part = freq_cum[idx0: idx0 + min_len]
    ewm_part = ewm[idx0: idx0 + min_len]
    X_dense = np.concatenate([X_main, freq_part, ewm_part], axis=1).astype(np.float32)

    # labels Y: próximo concurso do “t” atual
    Y_list: List[List[int]] = []
    for i in range(idx0, idx0 + min_len):
        prox = arr.iloc[i + 1].tolist()
        prox_limpo = [int(x) for x in prox if 0 <= int(x) <= max_d]
        Y_list.append(sorted(set(prox_limpo)))

    mlb = MultiLabelBinarizer(classes=list(range(max_d + 1)))
    Y_bin = mlb.fit_transform(Y_list).astype(np.int8)

    if _HAS_SCIPY:
        X = sparse.csr_matrix(X_dense)  # mantém compatibilidade geral; models dense-only já convertem
        return X, Y_bin, max_d
    return X_dense, Y_bin, max_d


# =========================
# avaliação / scores
# =========================
def _avaliar_modelo_timeseries(clf: BaseEstimator, X: Any, Y: np.ndarray, n_splits: int) -> Tuple[float, float]:
    if getattr(Y, "size", 0) == 0:
        return 0.0, 0.0
    n = Y.shape[0]
    if n < max(12, n_splits + 2):
        return 0.0, 0.0

    tscv = TimeSeriesSplit(n_splits=n_splits)

    f1s: List[float] = []
    accs: List[float] = []

    for train_idx, test_idx in tscv.split(np.arange(n)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = Y[train_idx]
        y_test = Y[test_idx]

        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)

        f1s.append(f1_score(y_test, pred, average="micro", zero_division=0))
        accs.append(accuracy_score(y_test, pred))

    return float(np.mean(f1s)), float(np.mean(accs))


def _scores_ultima_amostra(modelo: BaseEstimator, X: Any, max_d: int) -> np.ndarray:
    """
    Retorna um vetor de “scores” por dezena (0..max_d).
    Tenta predict_proba; se não rolar, decision_function; se não, fallback random.
    """
    try:
        proba = modelo.predict_proba(X)
        # MultiOutputClassifier / multi-label pode retornar list[ (n,2) ] por coluna
        if isinstance(proba, list):
            # pega P(y=1) de cada output
            last = []
            for p in proba:
                p = np.asarray(p)
                if p.ndim == 2 and p.shape[1] >= 2:
                    last.append(float(p[-1, 1]))
                else:
                    last.append(float(p[-1]))
            scores = np.asarray(last, dtype=np.float32)
            # scores aqui é por output (n_outputs == n_classes). Precisamos alinhar com 0..max_d.
            # Se modelo treinou com Y de tamanho max_d+1, ok.
            if scores.shape[0] == max_d + 1:
                return scores
            # tenta pad/trim
            out = np.zeros((max_d + 1,), dtype=np.float32)
            m = min(out.shape[0], scores.shape[0])
            out[:m] = scores[:m]
            return out

        proba = np.asarray(proba)
        if proba.ndim == 2:
            scores = proba[-1]
            if scores.shape[0] == max_d + 1:
                return scores.astype(np.float32, copy=False)
            out = np.zeros((max_d + 1,), dtype=np.float32)
            m = min(out.shape[0], scores.shape[0])
            out[:m] = scores[:m]
            return out
    except Exception:
        pass

    try:
        df = modelo.decision_function(X)
        df = np.asarray(df)
        if df.ndim == 2:
            s = df[-1]
        else:
            s = df
        s = s.astype(np.float32, copy=False)
        out = np.zeros((max_d + 1,), dtype=np.float32)
        m = min(out.shape[0], s.shape[0])
        out[:m] = s[:m]
        # normaliza pra 0..1 via sigmoid (só pra ranking)
        out = 1.0 / (1.0 + np.exp(-out))
        return out
    except Exception:
        pass

    rng = np.random.default_rng(CFG.random_state)
    return rng.random(max_d + 1, dtype=np.float32)


# =========================
# XGBoost (GPU se existir)
# =========================
def _criar_xgb_classifier() -> Optional[Any]:
    """
    Retorna XGBClassifier configurado.
    Se xgboost não estiver instalado, retorna None.
    Usa GPU se suportado; senão CPU.
    """
    try:
        import xgboost as xgb  # type: ignore
    except Exception:
        return None

    # heurística simples: tenta CUDA primeiro (xgboost >=2 costuma aceitar device='cuda')
    # se falhar na hora do fit, o _rank_por_modelo faz fallback automático.
    return xgb.XGBClassifier(
        n_estimators=3500,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        reg_alpha=0.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        tree_method="hist",
        device="cuda",
        n_jobs=-1,
        random_state=CFG.random_state,
        eval_metric="logloss",
    )


# =========================
# ranking por modelo (com tuning pesado)
# =========================
def _pipeline_padrao(base_estimator: BaseEstimator, sparse_ok: bool = True) -> Pipeline:
    steps = []
    # scaler que não destrói sparse
    if sparse_ok:
        steps.append(("scaler", MaxAbsScaler()))
    steps.append(("clf", base_estimator))
    return Pipeline(steps=steps)


def _tuning_param_space(nome_modelo: str) -> Optional[dict]:
    if nome_modelo == "random_forest":
        return {
            "clf__n_estimators": [800, 1200, 1600],
            "clf__max_depth": [None, 20, 35, 50],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", 0.4, 0.6],
        }
    if nome_modelo == "extra_trees":
        return {
            "clf__n_estimators": [1200, 1800, 2400],
            "clf__max_depth": [None, 25, 40, 60],
            "clf__min_samples_leaf": [1, 2, 3],
            "clf__max_features": ["sqrt", "log2", 0.4, 0.7],
        }
    if nome_modelo == "logistic_regression":
        return {
            "clf__estimator__C": [0.2, 0.5, 1.0, 2.0, 5.0],
            "clf__estimator__penalty": ["l2"],
        }
    if nome_modelo == "hist_gradient_boosting":
        return {
            "clf__estimator__max_iter": [400, 700, 1000],
            "clf__estimator__learning_rate": [0.03, 0.05, 0.08],
            "clf__estimator__max_depth": [None, 6, 9],
            "clf__estimator__l2_regularization": [0.0, 0.1, 0.5],
        }
    if nome_modelo == "xgboost":
        return {
            "clf__estimator__max_depth": [5, 7, 9],
            "clf__estimator__learning_rate": [0.02, 0.03, 0.05],
            "clf__estimator__subsample": [0.75, 0.85, 0.95],
            "clf__estimator__colsample_bytree": [0.75, 0.85, 0.95],
            "clf__estimator__n_estimators": [2200, 3500, 4500],
            "clf__estimator__reg_lambda": [1.0, 1.5, 2.0],
        }
    return None


def _rank_por_modelo(
    df: pd.DataFrame,
    top_n: int,
    base_estimator: BaseEstimator,
    tipo_loteria: str,
    nome_modelo: str,
) -> Tuple[List[int], float]:
    X, Y, max_d = _preparar_ml_pesado(df, tipo_loteria)

    # pouca amostra -> heurística para não cair em sequência
    if getattr(Y, "size", 0) == 0 or (hasattr(X, "shape") and X.shape[0] < 30):
        return predizer_por_frequencia(df, top_n), 0.0

    # carrega salvo
    salvo = _tentar_carregar_modelo(tipo_loteria, nome_modelo)
    pipeline_antigo = salvo.get("pipeline") if isinstance(salvo, dict) else None
    n_features_salvo = salvo.get("n_features") if isinstance(salvo, dict) else None
    lags_salvo = salvo.get("lags") if isinstance(salvo, dict) else None

    melhor_pipeline: Optional[BaseEstimator] = None
    melhor_score = 0.0

    # avalia antigo
    if pipeline_antigo is not None and n_features_salvo == X.shape[1] and lags_salvo == CFG.lags:
        try:
            f1_old, acc_old = _avaliar_modelo_timeseries(pipeline_antigo, X, Y, CFG.cv_splits)
            melhor_score = (f1_old + acc_old) / 2.0
            melhor_pipeline = pipeline_antigo
        except Exception:
            melhor_score = 0.0
            melhor_pipeline = None

    # monta pipeline novo
    # (para multi-label, usamos OneVsRest nos que não suportam multioutput bem)
    pipeline_novo: BaseEstimator

    # Se base_estimator já suporta multioutput naturalmente (RF/ET), dá pra treinar direto com Y.
    # Ainda assim, OneVsRest costuma dar probabilidades mais fáceis e permite tuning consistente.
    # Aqui: RF/ET direto; demais via OVR.
    if nome_modelo in ("random_forest", "extra_trees"):
        pipeline_novo = _pipeline_padrao(base_estimator, sparse_ok=True)
    else:
        pipeline_novo = _pipeline_padrao(OneVsRestClassifier(base_estimator, n_jobs=-1), sparse_ok=True)

    # tuning pesado (usa CPU de verdade)
    pipeline_candidato = pipeline_novo
    if X.shape[0] >= CFG.use_random_search_min_samples:
        space = _tuning_param_space(nome_modelo)
        if space:
            try:
                # TimeSeriesSplit não é suportado direto pelo RandomizedSearchCV para multilabel em alguns casos;
                # usamos um CV “índices” com TimeSeriesSplit via split manual.
                cv = list(TimeSeriesSplit(n_splits=CFG.cv_splits).split(np.arange(Y.shape[0])))

                search = RandomizedSearchCV(
                    estimator=pipeline_novo,
                    param_distributions=space,
                    n_iter=CFG.randomized_search_iter,
                    scoring="f1_micro",
                    n_jobs=-1,
                    cv=cv,
                    random_state=CFG.random_state,
                    verbose=0,
                )
                search.fit(X, Y)
                pipeline_candidato = search.best_estimator_
            except Exception:
                pipeline_candidato = pipeline_novo

    # avalia candidato
    try:
        f1_new, acc_new = _avaliar_modelo_timeseries(pipeline_candidato, X, Y, CFG.cv_splits)
        score_novo = (f1_new + acc_new) / 2.0
    except Exception:
        score_novo = 0.0

    if score_novo >= melhor_score or melhor_pipeline is None:
        melhor_score = score_novo
        melhor_pipeline = pipeline_candidato

    # treina final em tudo
    # fallback extra: XGBoost GPU pode falhar se não houver CUDA; tenta CPU automaticamente
    try:
        melhor_pipeline.fit(X, Y)
    except Exception:
        if nome_modelo == "xgboost":
            # força CPU
            try:
                import xgboost as xgb  # type: ignore
                cpu_est = xgb.XGBClassifier(
                    n_estimators=3500,
                    max_depth=7,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=1.5,
                    reg_alpha=0.0,
                    min_child_weight=1.0,
                    objective="binary:logistic",
                    tree_method="hist",
                    device="cpu",
                    n_jobs=-1,
                    random_state=CFG.random_state,
                    eval_metric="logloss",
                )
                melhor_pipeline = _pipeline_padrao(OneVsRestClassifier(cpu_est, n_jobs=-1), sparse_ok=True)
                melhor_pipeline.fit(X, Y)
            except Exception:
                # se ainda falhar, cai no antigo (se existir) ou heurística
                if pipeline_antigo is not None:
                    melhor_pipeline = pipeline_antigo
                else:
                    return predizer_por_frequencia(df, top_n), 0.0
        else:
            if pipeline_antigo is not None:
                melhor_pipeline = pipeline_antigo
            else:
                return predizer_por_frequencia(df, top_n), 0.0

    # salva
    _salvar_modelo(
        tipo_loteria,
        nome_modelo,
        {
            "pipeline": melhor_pipeline,
            "n_features": int(X.shape[1]),
            "max_num": int(max_d),
            "n_samples": int(getattr(X, "shape", [0])[0]),
            "lags": int(CFG.lags),
            "cfg": {
                "lags": CFG.lags,
                "ewm_alpha": CFG.ewm_alpha,
                "cv_splits": CFG.cv_splits,
                "randomized_search_iter": CFG.randomized_search_iter,
            },
        },
    )

    # scores para a última amostra -> ranking
    scores = _scores_ultima_amostra(melhor_pipeline, X, max_d)

    # descarta 0
    scores_sem_zero = np.asarray(scores[1:], dtype=np.float32)
    idx = np.argsort(scores_sem_zero)[-top_n:][::-1]
    dezenas = [int(i + 1) for i in idx]
    dezenas = sorted(dezenas)[:top_n]

    return dezenas, float(melhor_score)


# =========================
# modelos (agora com +1 e GPU opcional)
# =========================
def predizer_random_forest(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    clf = RandomForestClassifier(
        n_estimators=1400,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=CFG.random_state,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "random_forest")


def predizer_extra_trees(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    clf = ExtraTreesClassifier(
        n_estimators=2200,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=CFG.random_state,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "extra_trees")


def predizer_logistic(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    # saga usa multi-core e funciona bem com sparse
    clf = LogisticRegression(
        max_iter=6000,
        solver="saga",
        penalty="l2",
        n_jobs=-1,
        random_state=CFG.random_state,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "logistic_regression")


def predizer_knn(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    # KNN é caro em predição; aqui fica mais “pesado” mas ainda realista
    clf = KNeighborsClassifier(
        n_neighbors=11,
        weights="distance",
        metric="minkowski",
        p=2,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "k_nearest_neighbors")


def predizer_hgb(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    # substitui o GradientBoosting (single-core) por HistGradientBoosting (bem mais forte)
    clf = HistGradientBoostingClassifier(
        max_iter=900,
        learning_rate=0.05,
        max_depth=9,
        l2_regularization=0.1,
        random_state=CFG.random_state,
    )
    return _rank_por_modelo(df, top_n, clf, tipo_loteria, "hist_gradient_boosting")


def predizer_xgboost(df: pd.DataFrame, top_n: int, tipo_loteria: str) -> Tuple[List[int], float]:
    xgb_clf = _criar_xgb_classifier()
    if xgb_clf is None:
        # se não tem xgboost instalado, cai em ExtraTrees (forte) sem quebrar API
        return predizer_extra_trees(df, top_n, tipo_loteria)
    return _rank_por_modelo(df, top_n, xgb_clf, tipo_loteria, "xgboost")


# =========================
# orquestra
# =========================
def gerar_palpite(df: pd.DataFrame, tipo_loteria: str) -> Dict[str, Any]:
    """
    Observação objetiva: loteria é aleatória; isso aqui é experimento estatístico/ML,
    não aumenta garantia de acerto.
    """
    top_n = _top_n_por_tipo(tipo_loteria)
    historico = _carregar_historico()

    # heurísticas
    freq = predizer_por_frequencia(df, top_n)
    rec = predizer_por_recencia(df, top_n)

    # modelos pesados (CPU) + xgboost (GPU se existir)
    rf, rf_sc = predizer_random_forest(df, top_n, tipo_loteria)
    et, et_sc = predizer_extra_trees(df, top_n, tipo_loteria)
    lg, lg_sc = predizer_logistic(df, top_n, tipo_loteria)
    kn, kn_sc = predizer_knn(df, top_n, tipo_loteria)
    hgb, hgb_sc = predizer_hgb(df, top_n, tipo_loteria)
    xgb, xgb_sc = predizer_xgboost(df, top_n, tipo_loteria)

    resultados: Dict[str, List[int]] = {
        "frequencia_simples": freq,
        "recencia_ponderada": rec,
        "random_forest": rf,
        "extra_trees": et,
        "logistic_regression": lg,
        "k_nearest_neighbors": kn,
        "hist_gradient_boosting": hgb,
        "xgboost": xgb,
    }

    desempenho = {
        "random_forest": rf_sc,
        "extra_trees": et_sc,
        "logistic_regression": lg_sc,
        "k_nearest_neighbors": kn_sc,
        "hist_gradient_boosting": hgb_sc,
        "xgboost": xgb_sc,
    }

    # pesos: mistura desempenho atual + histórico
    def _mix(nome: str, base: float) -> float:
        h = float(historico.get(nome, 1.0))
        a = float(desempenho.get(nome, 0.0))
        return base + (a + h) / 2.0

    pesos_modelos: Dict[str, float] = {
        "frequencia_simples": 0.55,
        "recencia_ponderada": 0.95,
        "random_forest": _mix("random_forest", 1.10),
        "extra_trees": _mix("extra_trees", 1.20),
        "logistic_regression": _mix("logistic_regression", 0.95),
        "k_nearest_neighbors": _mix("k_nearest_neighbors", 0.85),
        "hist_gradient_boosting": _mix("hist_gradient_boosting", 1.15),
        "xgboost": _mix("xgboost", 1.25),
    }

    votos: Counter[int] = Counter()
    for nome, lista in resultados.items():
        peso = float(pesos_modelos.get(nome, 1.0))
        for d in lista:
            votos[int(d)] += peso

    melhores = sorted([d for d, _ in votos.most_common(top_n)])

    resultados["melhor_combinacao"] = melhores
    resultados["avaliacao_modelos"] = {k: float(round(v, 4)) for k, v in desempenho.items()}

    _salvar_historico(tipo_loteria, desempenho)

    # inclui metadados de treino “pra debug”
    resultados_meta: Dict[str, Any] = {
        "cfg": {
            "lags": CFG.lags,
            "ewm_alpha": CFG.ewm_alpha,
            "cv_splits": CFG.cv_splits,
            "randomized_search_iter": CFG.randomized_search_iter,
            "random_search_min_samples": CFG.use_random_search_min_samples,
        }
    }

    return {**resultados, **resultados_meta}
