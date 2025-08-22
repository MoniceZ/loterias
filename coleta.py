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
    return 20


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
                # Título geralmente é o número do concurso no <strong>
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

        except Exception as e:
            self.log.emit(f"❌ Erro na coleta: {e}")
            self.finalizado.emit(pd.DataFrame())
        finally:
            if driver is not None:
                driver.quit()
