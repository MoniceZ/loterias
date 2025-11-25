from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def gerar_url(loteria_nome: str) -> str:
    slug = (
        loteria_nome.lower()
        .replace("+", "mais")
        .replace(" ", "-")
        .replace("á", "a").replace("é", "e").replace("í", "i")
        .replace("ó", "o").replace("ú", "u")
        .replace("ç", "c")
    )
    return f"https://asloterias.com.br/lista-de-resultados-da-{slug}"


def obter_max_dezenas(loteria_nome: str) -> int:
    loteria = loteria_nome.lower()
    if "facil" in loteria:
        return 15
    if "mega" in loteria:
        return 6
    if "quina" in loteria:
        return 5
    if "lotomania" in loteria:
        return 20
    return 20


def calcular_estatisticas_loteria(df: pd.DataFrame, loteria_nome: str) -> pd.DataFrame:
    """
    Calcula estatísticas das dezenas com base no DataFrame retornado pela coleta.

    Saída: DataFrame com colunas, por dezena:
      - dezena
      - qtd_sorteios
      - freq_rel
      - esperado
      - dif_abs
      - dif_pct
      - atraso_atual
      - max_atraso
      - primeiro_concurso
      - ultimo_concurso
      - media_intervalo
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "dezena",
                "qtd_sorteios",
                "freq_rel",
                "esperado",
                "dif_abs",
                "dif_pct",
                "atraso_atual",
                "max_atraso",
                "primeiro_concurso",
                "ultimo_concurso",
                "media_intervalo",
            ]
        )

    cols_dezenas = sorted([c for c in df.columns if c.startswith("num_")])
    if not cols_dezenas:
        raise ValueError("DataFrame sem colunas de dezenas (num_XX).")

    # Série com todas as dezenas sorteadas
    matriz = df[cols_dezenas].to_numpy().ravel()
    numeros = pd.Series(matriz[~pd.isna(matriz)], dtype=int)

    loteria = loteria_nome.lower()
    # Define universo de dezenas conforme o tipo da loteria
    if "facil" in loteria:
        universo = list(range(1, 26))
    elif "mega" in loteria:
        universo = list(range(1, 61))
    elif "quina" in loteria:
        universo = list(range(1, 81))
    elif "lotomania" in loteria:
        universo = list(range(0, 100))
    else:
        universo = list(range(int(numeros.min()), int(numeros.max()) + 1))

    universo_set = set(universo)

    # Frequência absoluta
    freq = numeros.value_counts().reindex(universo, fill_value=0)

    total_concursos = int(len(df))
    dezenas_por_concurso = int(len(cols_dezenas))
    total_sorteios = int(total_concursos * dezenas_por_concurso)
    qtd_numeros = int(len(universo))
    esperado = total_sorteios / qtd_numeros if qtd_numeros > 0 else 0.0

    df_freq = pd.DataFrame(
        {
            "dezena": [int(x) for x in freq.index],
            "qtd_sorteios": freq.values.astype(int),
        }
    )

    # Frequência relativa e desvio do esperado
    df_freq["freq_rel"] = df_freq["qtd_sorteios"] / float(total_sorteios) if total_sorteios > 0 else 0.0
    df_freq["esperado"] = esperado
    df_freq["dif_abs"] = df_freq["qtd_sorteios"] - df_freq["esperado"]
    df_freq["dif_pct"] = df_freq["dif_abs"] / df_freq["esperado"] if esperado > 0 else 0.0

    # Cálculo de atrasos e intervalos
    df_sorted = df.sort_values("concurso").reset_index(drop=True)
    ultimo_concurso = int(df_sorted["concurso"].max())

    hits_count: Dict[int, int] = {n: 0 for n in universo}
    first_hit: Dict[int, int | None] = {n: None for n in universo}
    last_hit: Dict[int, int | None] = {n: None for n in universo}
    current_run: Dict[int, int] = {n: 0 for n in universo}
    max_run: Dict[int, int] = {n: 0 for n in universo}

    for _, row in df_sorted.iterrows():
        conc = int(row["concurso"])
        presentes = set(int(v) for v in row[cols_dezenas].to_list())
        presentes &= universo_set

        for n in universo:
            if n in presentes:
                # fechou uma sequência de atraso
                if current_run[n] > max_run[n]:
                    max_run[n] = current_run[n]
                current_run[n] = 0

                hits_count[n] += 1
                if first_hit[n] is None:
                    first_hit[n] = conc
                last_hit[n] = conc
            else:
                current_run[n] += 1

    # Ajuste final das sequências em aberto
    for n in universo:
        if current_run[n] > max_run[n]:
            max_run[n] = current_run[n]

    atraso_atual: Dict[int, float | pd._libs.missing.NAType] = {}
    media_intervalo: Dict[int, float | pd._libs.missing.NAType] = {}

    for n in universo:
        if hits_count[n] > 0:
            # atraso atual = quantos concursos desde a última vez que saiu
            atraso_atual[n] = ultimo_concurso - int(last_hit[n])
        else:
            atraso_atual[n] = pd.NA

        if hits_count[n] > 1 and first_hit[n] is not None:
            # média de intervalo entre saídas, em concursos
            media_intervalo[n] = (ultimo_concurso - int(first_hit[n])) / float(hits_count[n] - 1)
        else:
            media_intervalo[n] = pd.NA

    df_freq["atraso_atual"] = df_freq["dezena"].map(atraso_atual)
    df_freq["max_atraso"] = df_freq["dezena"].map(max_run)
    df_freq["primeiro_concurso"] = df_freq["dezena"].map(first_hit)
    df_freq["ultimo_concurso"] = df_freq["dezena"].map(last_hit)
    df_freq["media_intervalo"] = df_freq["dezena"].map(media_intervalo)

    # Ordena por atraso atual decrescente como default (mais atrasados primeiro)
    df_freq = df_freq.sort_values(["atraso_atual", "dezena"], ascending=[False, True]).reset_index(drop=True)

    return df_freq


@dataclass(frozen=True)
class ColetaConfig:
    headless: bool = True
    timeout_s: int = 15


def _criar_driver(cfg: ColetaConfig) -> webdriver.Chrome:
    opts = Options()
    if cfg.headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--window-size=1200,900")
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)


class ColetorThread(QThread):
    progresso = pyqtSignal(int)
    log = pyqtSignal(str)
    finalizado = pyqtSignal(pd.DataFrame)
    estatisticas = pyqtSignal(pd.DataFrame)  # << novo sinal com os cálculos

    def __init__(self, loteria_nome: str, quantidade: int, cfg: ColetaConfig | None = None):
        super().__init__()
        self.loteria_nome = loteria_nome
        self.quantidade = int(quantidade)
        self.cfg = cfg or ColetaConfig()
        self.url = gerar_url(loteria_nome)
        self.max_dezenas = obter_max_dezenas(loteria_nome)

    def run(self) -> None:
        self.log.emit("🟡 Inicializando WebDriver...")
        driver = None
        try:
            driver = _criar_driver(self.cfg)
            self.log.emit(f"🌐 Acessando URL: {self.url}")
            driver.get(self.url)

            wait = WebDriverWait(driver, self.cfg.timeout_s)
            wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "strong")))
            elementos = driver.find_elements(By.TAG_NAME, "strong")
            self.log.emit(f"🔎 {len(elementos)} concursos encontrados na página.")

            resultados: List[Dict[str, str | int]] = []
            rx_data = re.compile(r"\d{2}/\d{2}/\d{4}")
            rx_dezenas = re.compile(r"\d{2}")

            total_alvo = max(self.quantidade, 1)
            for idx, strong in enumerate(elementos, start=1):
                concurso = strong.text.strip()
                if not concurso.isdigit():
                    continue

                texto = driver.execute_script(
                    """
                    const el = arguments[0];
                    let node = el.nextSibling;
                    while (node && node.nodeType !== Node.TEXT_NODE) node = node.nextSibling;
                    return node ? node.textContent.trim() : "";
                    """,
                    strong,
                )
                if not texto:
                    continue

                m_data = rx_data.search(texto)
                dezenas = rx_dezenas.findall(texto)

                if not m_data or len(dezenas) < self.max_dezenas:
                    continue

                data = m_data.group(0)
                ultimas = dezenas[-self.max_dezenas:]
                row = {"concurso": int(concurso), "data": data}
                for j, dez in enumerate(ultimas, 1):
                    row[f"num_{j:02d}"] = int(dez)

                resultados.append(row)

                progresso = int(min(100, (len(resultados) / total_alvo) * 100))
                self.progresso.emit(progresso)
                self.log.emit(f"✅ Concurso {concurso} coletado.")

                if len(resultados) >= total_alvo:
                    break

            df = pd.DataFrame(resultados)
            self.finalizado.emit(df)

            # Cálculos mais fortes sobre as dezenas
            try:
                df_stats = calcular_estatisticas_loteria(df, self.loteria_nome)
                self.estatisticas.emit(df_stats)
                self.log.emit("📊 Estatísticas das dezenas calculadas com sucesso.")
            except Exception as e_calc:
                self.log.emit(f"⚠️ Erro ao calcular estatísticas: {e_calc}")

        except Exception as e:
            self.log.emit(f"❌ Erro na coleta: {e}")
            self.finalizado.emit(pd.DataFrame())
            # Em caso de erro, manda DataFrame vazio também nas estatísticas
            self.estatisticas.emit(pd.DataFrame())
        finally:
            if driver is not None:
                driver.quit()
