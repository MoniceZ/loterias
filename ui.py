from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QProgressBar,
    QTextEdit,
    QMessageBox,
    QWidget,
)

from coleta import ColetorThread
from predicao import carregar_dados, gerar_palpite

LOTERIAS = ["Mega Sena", "Lotofacil", "Quina", "Lotomania"]


class App(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Coletar Resultados de Loterias")
        self.setFixedSize(520, 520)
        self.setWindowIcon(QIcon("loteria.ico"))
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Loteria:"))
        self.combo_loteria = QComboBox()
        self.combo_loteria.addItems(LOTERIAS)
        layout.addWidget(self.combo_loteria)

        layout.addWidget(QLabel("Quantidade de concursos:"))
        self.input_qtd = QLineEdit("10")
        self.input_qtd.setMaxLength(4)
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

    # ---------- coleta ----------
    def iniciar_coleta(self) -> None:
        loteria = self.combo_loteria.currentText()
        qtd_txt = self.input_qtd.text().strip()

        if not qtd_txt.isdigit() or int(qtd_txt) <= 0:
            QMessageBox.critical(self, "Erro", "Quantidade inválida.")
            return

        self.log_area.clear()
        self.progress.setValue(0)
        self.log_area.append("🚀 Iniciando coleta...")

        self.thread = ColetorThread(loteria_nome=loteria, quantidade=int(qtd_txt))
        self.thread.progresso.connect(self.progress.setValue)
        self.thread.log.connect(self.log_area.append)
        self.thread.finalizado.connect(self._finalizar_coleta)
        self.thread.start()

    def _finalizar_coleta(self, df: pd.DataFrame) -> None:
        if df.empty:
            self.log_area.append("⚠️ Nenhum resultado encontrado.")
            QMessageBox.warning(self, "Aviso", "Nenhum resultado foi coletado.")
            return

        nome_base = self.combo_loteria.currentText().replace(" ", "_").replace("+", "mais")
        saida = Path(f"{nome_base}_resultados.csv")
        df.to_csv(saida, index=False, encoding="utf-8-sig")
        self.log_area.append(f"💾 {len(df)} concursos salvos em {saida.name}")
        QMessageBox.information(self, "Sucesso", f"{len(df)} concursos salvos com sucesso.")

    # ---------- predição ----------
    def predizer_jogo(self) -> None:
        loteria = self.combo_loteria.currentText()
        nome_base = loteria.replace(" ", "_").replace("+", "mais")
        arquivo_csv = Path(f"{nome_base}_resultados.csv")

        if not arquivo_csv.exists():
            QMessageBox.critical(self, "Erro", "Você precisa coletar os dados primeiro.")
            return

        try:
            self.log_area.append("🔍 Iniciando predição...")
            self.progress.setValue(0)

            df = carregar_dados(arquivo_csv)
            palpites = gerar_palpite(df, loteria)

            etapas = [
                "frequencia_simples",
                "recencia_ponderada",
                "random_forest",
                "logistic_regression",
                "k_nearest_neighbors",
                "gradient_boosting",
                "melhor_combinacao",
            ]

            for i, chave in enumerate(etapas, start=1):
                nums = palpites.get(chave, [])
                self.log_area.append(f"{'🧠 Voto final' if chave=='melhor_combinacao' else '•'} {chave}: {', '.join(map(str, nums))}")
                self.progress.setValue(int(i / len(etapas) * 100))

            QMessageBox.information(
                self,
                "Previsão Concluída",
                f"Melhor jogo previsto: {', '.join(map(str, palpites['melhor_combinacao']))}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Ocorreu um erro na predição: {e}")
