# predicao.py
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from itertools import combinations

# --- ETAPA 1: Pré-processamento ---
def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)
    dezenas_cols = [col for col in df.columns if col.startswith("num_")]
    df[dezenas_cols] = df[dezenas_cols].astype(int)
    return df

# --- ETAPA 2: Frequência Simples ---
def predizer_por_frequencia(df: pd.DataFrame, top_n: int) -> list:
    dezenas = df[[col for col in df.columns if col.startswith("num_")]].values.flatten()
    contagem = Counter(dezenas)
    return sorted([int(num) for num, _ in contagem.most_common(top_n)])

# --- ETAPA 3: Recência Ponderada ---
def predizer_por_recencia(df: pd.DataFrame, top_n: int) -> list:
    col_dezenas = [col for col in df.columns if col.startswith("num_")]
    pesos = np.linspace(1, 0.1, num=len(df))
    contagem = Counter()
    for i, row in enumerate(df[col_dezenas].values):
        for dezena in row:
            contagem[dezena] += pesos[i]
    return sorted([int(num) for num, _ in contagem.most_common(top_n)])

# --- ETAPA 4: Preparação dos Dados para ML ---
def preparar_dados_para_ml(df: pd.DataFrame):
    dezenas_cols = [col for col in df.columns if col.startswith("num_")]
    df_dezenas = df[dezenas_cols].astype(int)
    max_dezena = df_dezenas.max().max()
    X = []
    Y = []
    for i in range(len(df_dezenas) - 1):
        atual = df_dezenas.iloc[i].tolist()
        proximo = df_dezenas.iloc[i + 1].tolist()
        linha = [0] * (max_dezena + 1)
        for dez in atual:
            linha[dez] = 1
        X.append(linha)
        Y.append(list(set(proximo)))
    mlb = MultiLabelBinarizer(classes=list(range(max_dezena + 1)))
    Y_bin = mlb.fit_transform(Y)
    return np.array(X), Y_bin, max_dezena

# --- ETAPA 5: Predição por Múltiplos Modelos Supervisionados ---
def predizer_modelo_classico(df: pd.DataFrame, top_n: int, modelo_class):
    X, Y, max_dezena = preparar_dados_para_ml(df)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    modelo = OneVsRestClassifier(modelo_class())
    modelo.fit(X_train, y_train)
    pred = modelo.predict([X[-1]])[0]
    indices = np.argsort(pred)[-top_n:][::-1]
    return sorted(indices)

def predizer_random_forest(df, top_n):
    return predizer_modelo_classico(df, top_n, lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1))

def predizer_logistic(df, top_n):
    return predizer_modelo_classico(df, top_n, lambda: LogisticRegression(max_iter=1000))

def predizer_knn(df, top_n):
    return predizer_modelo_classico(df, top_n, lambda: KNeighborsClassifier(n_neighbors=3))

def predizer_gb(df, top_n):
    return predizer_modelo_classico(df, top_n, lambda: GradientBoostingClassifier(n_estimators=100))

# --- FUNÇÃO FINAL: PALPITES COMPLETOS ---
def gerar_palpite(df: pd.DataFrame, tipo_loteria: str) -> dict:
    tipo = tipo_loteria.lower()
    top_n = 15 if "facil" in tipo else 6

    resultados = {
        "frequencia_simples": predizer_por_frequencia(df, top_n),
        "recencia_ponderada": predizer_por_recencia(df, top_n),
        "random_forest": predizer_random_forest(df, top_n),
        "logistic_regression": predizer_logistic(df, top_n),
        "k_nearest_neighbors": predizer_knn(df, top_n),
        "gradient_boosting": predizer_gb(df, top_n),
    }

    votos = Counter()
    for lista in resultados.values():
        for dez in lista:
            votos[dez] += 1
    melhor = sorted([num for num, _ in votos.most_common(top_n)])
    resultados["melhor_combinacao"] = melhor
    return resultados

# --- TESTE ISOLADO ---
if __name__ == "__main__":
    caminho = "Mega_Sena_resultados.csv"  # ou "Lotofacil_resultados.csv"
    tipo = "Mega Sena"
    df = carregar_dados(caminho)
    palpites = gerar_palpite(df, tipo)
    for metodo, nums in palpites.items():
        print(f"{metodo}: {nums}")
