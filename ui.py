import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QLabel, QComboBox, QLineEdit, QPushButton,
    QVBoxLayout, QProgressBar, QTextEdit, QMessageBox
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt
from coleta import ColetorThread, gerar_url
from predicao import gerar_palpite, carregar_dados

LOTERIAS = ["Mega Sena", "Lotofacil"]

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coletar Resultados de Loterias")
        self.setFixedSize(500, 500)
        self.setWindowIcon(QIcon("loteria.ico"))
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.label_loteria = QLabel("Loteria:")
        layout.addWidget(self.label_loteria)

        self.combo_loteria = QComboBox()
        self.combo_loteria.addItems(LOTERIAS)
        layout.addWidget(self.combo_loteria)

        self.label_qtd = QLabel("Quantidade de concursos:")
        layout.addWidget(self.label_qtd)

        self.input_qtd = QLineEdit("10")
        layout.addWidget(self.input_qtd)

        self.btn_coletar = QPushButton("Coletar Resultados")
        self.btn_coletar.clicked.connect(self.iniciar_coleta)
        layout.addWidget(self.btn_coletar)

        self.btn_predizer = QPushButton("Prever Jogo Ideal")
        self.btn_predizer.clicked.connect(self.predizer_jogo)
        layout.addWidget(self.btn_predizer)

        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def iniciar_coleta(self):
        loteria = self.combo_loteria.currentText()
        qtd_text = self.input_qtd.text()

        if not qtd_text.isdigit():
            QMessageBox.critical(self, "Erro", "Quantidade inválida.")
            return

        url = gerar_url(loteria)
        qtd = int(qtd_text)

        self.log_area.clear()
        self.progress.setValue(0)
        self.log_area.append("🚀 Iniciando coleta...")

        self.thread = ColetorThread(url, qtd)
        self.thread.progresso.connect(self.progress.setValue)
        self.thread.log.connect(self.log_area.append)
        self.thread.finalizado.connect(self.finalizar)
        self.thread.start()

    def finalizar(self, df: pd.DataFrame):
        if df.empty:
            self.log_area.append("⚠️ Nenhum resultado encontrado.")
            QMessageBox.warning(self, "Aviso", "Nenhum resultado foi coletado.")
        else:
            nome = self.combo_loteria.currentText().replace(" ", "_").replace("+", "mais")
            df.to_csv(f"{nome}_resultados.csv", index=False, encoding="utf-8-sig")
            self.log_area.append(f"💾 {len(df)} concursos salvos em {nome}_resultados.csv")
            QMessageBox.information(self, "Sucesso", f"{len(df)} concursos salvos com sucesso.")

    def predizer_jogo(self):
        loteria = self.combo_loteria.currentText()
        nome = loteria.replace(" ", "_").replace("+", "mais")
        arquivo_csv = f"{nome}_resultados.csv"

        try:
            df = carregar_dados(arquivo_csv)
            self.log_area.append("🔍 Iniciando predição...")
            self.progress.setValue(0)

            total_passos = 7
            passo = 0
            palpites = {}

            self.log_area.append("📊 Frequência simples...")
            palpites["frequencia_simples"] = gerar_palpite(df, loteria)["frequencia_simples"]
            passo += 1
            self.progress.setValue(int((passo / total_passos) * 100))

            self.log_area.append("📈 Recência ponderada...")
            palpites["recencia_ponderada"] = gerar_palpite(df, loteria)["recencia_ponderada"]
            passo += 1
            self.progress.setValue(int((passo / total_passos) * 100))

            self.log_area.append("🌲 Random Forest...")
            palpites["random_forest"] = gerar_palpite(df, loteria)["random_forest"]
            passo += 1
            self.progress.setValue(int((passo / total_passos) * 100))

            self.log_area.append("📦 Regressão Logística...")
            palpites["logistic_regression"] = gerar_palpite(df, loteria)["logistic_regression"]
            passo += 1
            self.progress.setValue(int((passo / total_passos) * 100))

            self.log_area.append("📡 KNN...")
            palpites["k_nearest_neighbors"] = gerar_palpite(df, loteria)["k_nearest_neighbors"]
            passo += 1
            self.progress.setValue(int((passo / total_passos) * 100))

            self.log_area.append("🚀 Gradient Boosting...")
            palpites["gradient_boosting"] = gerar_palpite(df, loteria)["gradient_boosting"]
            passo += 1
            self.progress.setValue(int((passo / total_passos) * 100))

            self.log_area.append("🧠 Votação final...")
            palpites["melhor_combinacao"] = gerar_palpite(df, loteria)["melhor_combinacao"]
            passo += 1
            self.progress.setValue(100)

            self.log_area.append("\n🔮 Resultado final:")
            for metodo, nums in palpites.items():
                nums_str = ', '.join(str(n) for n in nums)
                self.log_area.append(f"{metodo}: {nums_str}")

            QMessageBox.information(self, "Previsão Concluída", f"Melhor jogo previsto: {', '.join(map(str, palpites['melhor_combinacao']))}")

        except FileNotFoundError:
            QMessageBox.critical(self, "Erro", "Você precisa coletar os dados primeiro.")
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Ocorreu um erro na predição: {e}")
