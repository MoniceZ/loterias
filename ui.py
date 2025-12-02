from __future__ import annotations
from pathlib import Path
import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QComboBox,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QProgressBar,
    QTextEdit,
    QMessageBox,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
)

from coleta import ColetorThread
from predicao import carregar_dados, gerar_palpite, _carregar_historico

LOTERIAS = ["Mega Sena", "Lotofacil", "Quina", "Lotomania"]


# ---------- Thread de predição (para não travar o front) ----------
class PreditorThread(QThread):
    progresso = pyqtSignal(int)
    log = pyqtSignal(str)
    resultado = pyqtSignal(dict)

    def __init__(self, loteria: str, arquivo_csv: Path):
        super().__init__()
        self.loteria = loteria
        self.arquivo_csv = arquivo_csv

    def run(self):
        try:
            self.log.emit("🔍 Iniciando predição...")
            self.progresso.emit(0)
            df = carregar_dados(self.arquivo_csv)
            # aqui é uma “caixa preta”: o progresso interno é do modelo,
            # então marcamos início (0%) e fim (100%)
            palpites = gerar_palpite(df, self.loteria)
            self.progresso.emit(100)
            self.resultado.emit(palpites)
        except Exception as e:
            self.log.emit(f"❌ Erro na predição: {e}")


# ---------- Interface principal ----------
class App(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Previsor de Loterias Inteligente")
        self.setFixedSize(700, 640)
        self.setWindowIcon(QIcon("loteria.ico"))
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setSpacing(8)

        lbl_title = QLabel("🎯 Previsor de Loterias Inteligente")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #FFFFFF;")
        layout.addWidget(lbl_title)

        layout.addWidget(QLabel("Loteria:"))
        self.combo_loteria = QComboBox()
        self.combo_loteria.addItems(LOTERIAS)
        layout.addWidget(self.combo_loteria)

        layout.addWidget(QLabel("Quantidade de concursos:"))
        self.input_qtd = QLineEdit("10")
        self.input_qtd.setMaxLength(4)
        layout.addWidget(self.input_qtd)

        self.btn_coletar = QPushButton("📥 Coletar Resultados")
        self.btn_coletar.clicked.connect(self.iniciar_coleta)
        layout.addWidget(self.btn_coletar)

        self.btn_predizer = QPushButton("🧠 Prever Jogo Ideal")
        self.btn_predizer.clicked.connect(self.predizer_jogo)
        layout.addWidget(self.btn_predizer)

        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.setFormat("%p%")  # mostra o valor em %
        layout.addWidget(self.progress)

        lbl_log = QLabel("📄 Log de Execução:")
        layout.addWidget(lbl_log)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        layout.addWidget(self.log_area)

        lbl_perf = QLabel("📊 Desempenho dos Modelos (F1/Acc médios):")
        layout.addWidget(lbl_perf)

        # agora com 4 colunas
        self.table_perf = QTableWidget(0, 4)
        self.table_perf.setHorizontalHeaderLabels([
            "Modelo",
            "Histórico (%)",
            "Atual (%)",
            "Evolução (p.p. / %)"
        ])
        self.table_perf.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table_perf.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table_perf)

        self.setLayout(layout)
        self.setStyleSheet("""
            QPushButton { padding: 6px; font-weight: bold; }
            QTextEdit { background: #000000; color: #FFFFFF; font-family: Consolas; font-size: 13px; }
            QProgressBar { height: 35px; }
            QComboBox, QLineEdit { height: 26px; }
            QTableWidget { background: #1e1e1e; color: #ffffff; gridline-color: #444444; }
            QHeaderView::section { background-color: #333333; color: #ffffff; font-weight: bold; }
        """)

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
        self.thread.log.connect(self._append_log)
        self.thread.finalizado.connect(self._finalizar_coleta)
        self.thread.start()

    def _finalizar_coleta(self, df: pd.DataFrame) -> None:
        if df.empty:
            self._append_log("⚠️ Nenhum resultado encontrado.")
            QMessageBox.warning(self, "Aviso", "Nenhum resultado foi coletado.")
            return

        nome_base = self.combo_loteria.currentText().replace(" ", "_").replace("+", "mais")
        saida = Path(f"{nome_base}_resultados.csv")
        df.to_csv(saida, index=False, encoding="utf-8-sig")
        self._append_log(f"💾 {len(df)} concursos salvos em {saida.name}")
        QMessageBox.information(self, "Sucesso", f"{len(df)} concursos salvos com sucesso.")

    # ---------- predição ----------
    def predizer_jogo(self) -> None:
        loteria = self.combo_loteria.currentText()
        nome_base = loteria.replace(" ", "_").replace("+", "mais")
        arquivo_csv = Path(f"{nome_base}_resultados.csv")

        if not arquivo_csv.exists():
            QMessageBox.critical(self, "Erro", "Você precisa coletar os dados primeiro.")
            return

        self._append_log("🧩 Preparando predição em segundo plano...")
        self.progress.setValue(0)

        self.pred_thread = PreditorThread(loteria, arquivo_csv)
        self.pred_thread.log.connect(self._append_log)
        self.pred_thread.progresso.connect(self.progress.setValue)
        self.pred_thread.resultado.connect(self._exibir_resultado)
        self.pred_thread.start()

    # ---------- exibição ----------
    def _exibir_resultado(self, palpites: dict) -> None:
        self._append_log("✅ Predição concluída!\n")

        etapas = [
            "frequencia_simples",
            "recencia_ponderada",
            "random_forest",
            "logistic_regression",
            "k_nearest_neighbors",
            "gradient_boosting",
            "melhor_combinacao",
        ]

        for chave in etapas:
            nums = palpites.get(chave, [])
            if not nums:
                continue

            if chave == "melhor_combinacao":
                self._append_log(f"🎯 Melhor combinação: {', '.join(map(str, nums))}\n")
            else:
                self._append_log(f"• {chave}: {', '.join(map(str, nums))}")

        # desempenho e variação histórica em %
        self._atualizar_tabela_desempenho(palpites)

        melhor = palpites.get("melhor_combinacao", [])
        if melhor:
            QMessageBox.information(
                self,
                "Previsão Concluída",
                f"Melhor jogo previsto:\n{', '.join(map(str, melhor))}",
            )

    def _atualizar_tabela_desempenho(self, palpites: dict) -> None:
        historico = _carregar_historico()        # médias históricas (0–1)
        atual = palpites.get("avaliacao_modelos", {})  # scores atuais (0–1)

        self.table_perf.setRowCount(0)

        if not atual:
            self._append_log("ℹ️ Nenhuma métrica de desempenho retornada pelos modelos.")
            return

        melhor_modelo = None
        melhor_score_atual = -1.0
        melhor_hist = 0.0
        melhor_delta_pct_rel = 0.0

        # garantir ordem estável
        for modelo in sorted(atual.keys()):
            score_atual = float(atual.get(modelo, 0.0))
            score_hist = float(historico.get(modelo, 0.0))

            # conversão para %
            hist_pct = score_hist * 100.0
            atual_pct = score_atual * 100.0
            delta_pp = atual_pct - hist_pct  # pontos percentuais

            if hist_pct > 0:
                delta_pct_rel = (delta_pp / hist_pct) * 100.0
            else:
                delta_pct_rel = 0.0

            # ícone
            if score_hist == 0 and score_atual > 0:
                emoji = "🆕"
            elif delta_pp > 0:
                emoji = "🔼"
            elif delta_pp < 0:
                emoji = "🔽"
            else:
                emoji = "⏺️"

            linha = self.table_perf.rowCount()
            self.table_perf.insertRow(linha)

            item_modelo = QTableWidgetItem(modelo)
            item_hist = QTableWidgetItem(f"{hist_pct:.2f}%")
            item_atual = QTableWidgetItem(f"{atual_pct:.2f}%")
            item_evol = QTableWidgetItem(
                f"{emoji} {delta_pp:+.2f} p.p. / {delta_pct_rel:+.2f}%"
            )

            for item in (item_modelo, item_hist, item_atual, item_evol):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            # tooltip explicando
            item_evol.setToolTip(
                f"Histórico: {hist_pct:.2f}%\n"
                f"Atual: {atual_pct:.2f}%\n"
                f"Variação: {delta_pp:+.2f} p.p. ({delta_pct_rel:+.2f}%)"
            )

            self.table_perf.setItem(linha, 0, item_modelo)
            self.table_perf.setItem(linha, 1, item_hist)
            self.table_perf.setItem(linha, 2, item_atual)
            self.table_perf.setItem(linha, 3, item_evol)

            # tracking do melhor modelo atual
            if score_atual > melhor_score_atual:
                melhor_score_atual = score_atual
                melhor_modelo = modelo
                melhor_hist = score_hist
                melhor_delta_pct_rel = delta_pct_rel

        # resumo no log com progresso em %
        if melhor_modelo is not None:
            hist_pct = melhor_hist * 100.0
            atual_pct = melhor_score_atual * 100.0
            self._append_log(
                f"🏅 Melhor modelo no momento: {melhor_modelo} "
                f"→ Atual: {atual_pct:.2f}% | Histórico: {hist_pct:.2f}% "
                f"| Evolução relativa: {melhor_delta_pct_rel:+.2f}%"
            )

    # ---------- log helper ----------
    def _append_log(self, msg: str) -> None:
        self.log_area.append(msg)
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)


# ---------- execução direta ----------
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    janela = App()
    janela.show()
    sys.exit(app.exec())
