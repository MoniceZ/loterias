import re
import time
import pandas as pd
from PyQt6.QtCore import QThread, pyqtSignal
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
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

def obter_max_dezenas(loteria: str) -> int:
    loteria = loteria.lower()
    if "facil" in loteria:
        return 15
    elif "mega" in loteria:
        return 6
    else:
        return 20

class ColetorThread(QThread):
    progresso = pyqtSignal(int)
    log = pyqtSignal(str)
    finalizado = pyqtSignal(pd.DataFrame)

    def __init__(self, url, quantidade):
        super().__init__()
        self.url = url
        self.quantidade = quantidade
        self.max_dezenas = obter_max_dezenas(url)

    def run(self):
        self.log.emit("🟡 Inicializando WebDriver...")
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        self.log.emit(f"🌐 Acessando URL: {self.url}")
        driver.get(self.url)
        time.sleep(2)

        elementos = driver.find_elements(By.TAG_NAME, "strong")
        self.log.emit(f"🔎 {len(elementos)} concursos encontrados.")

        resultados = []

        for i, strong in enumerate(elementos):
            try:
                concurso = strong.text.strip()
                if not concurso.isdigit():
                    continue

                texto = driver.execute_script("""
                    let el = arguments[0];
                    let node = el.nextSibling;
                    while (node && node.nodeType !== Node.TEXT_NODE) {
                        node = node.nextSibling;
                    }
                    return node ? node.textContent.trim() : "";
                """, strong)

                if not texto:
                    continue

                data_match = re.search(r"\d{2}/\d{2}/\d{4}", texto)
                dezenas_match = re.findall(r"\d{2}", texto)

                if not data_match or len(dezenas_match) < self.max_dezenas:
                    continue

                data = data_match.group(0)
                dezenas = dezenas_match[-self.max_dezenas:]

                resultado = {"concurso": concurso, "data": data}
                for j, dezena in enumerate(dezenas, start=1):
                    resultado[f"num_{j:02d}"] = dezena

                resultados.append(resultado)
                self.progresso.emit(int((len(resultados) / self.quantidade) * 100))
                self.log.emit(f"✅ Concurso {concurso} coletado.")

                if len(resultados) >= self.quantidade:
                    break

            except Exception as e:
                self.log.emit(f"⚠️ Erro no concurso {i + 1}: {e}")
                continue

        driver.quit()
        df = pd.DataFrame(resultados)
        self.finalizado.emit(df)
