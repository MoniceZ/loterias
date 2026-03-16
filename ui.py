from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from coleta import ColetorThread
from predicao import carregar_dados, gerar_palpite, _carregar_historico

# --- Atualizado: inclui Super Sete ---
LOTERIAS = ["Mega Sena", "Lotofacil", "Quina", "Lotomania", "Super Sete"]

DEFAULT_DEZENAS_POR_LOTERIA = {
    "Mega Sena": 6,
    "Lotofacil": 15,
    "Quina": 5,
    "Lotomania": 50,
    "Super Sete": 7,
}

LIMITES_DEZENAS_POR_LOTERIA = {
    "Mega Sena": (6, 20),
    "Lotofacil": (15, 20),
    "Quina": (5, 15),
    "Lotomania": (50, 50),
    "Super Sete": (7, 7),
}


def _is_super_sete(loteria: str) -> bool:
    t = (loteria or "").lower()
    return ("super" in t) and ("sete" in t)


def _format_dezenas(loteria: str, dezenas: List[int]) -> str:
    if "lotomania" in loteria.lower():
        return ", ".join(f"{int(d):02d}" for d in dezenas)
    if _is_super_sete(loteria):
        # Super Sete: dígitos 0..9 (mostrar como 0..9, sem zero à esquerda obrigatório)
        return " ".join(str(int(d)) for d in dezenas)
    return ", ".join(str(int(d)) for d in dezenas)


def _format_jogo(loteria: str, jogo: List[int]) -> str:
    return _format_dezenas(loteria, jogo)


def _ajustar_entrada_aposta(
    loteria: str,
    n_jogos: int,
    n_dezenas: int | None,
) -> Tuple[int, int | None, List[str]]:
    avisos: List[str] = []

    if n_jogos < 1:
        avisos.append("n_jogos ajustado para 1 (mínimo permitido).")
        n_jogos = 1

    limites = LIMITES_DEZENAS_POR_LOTERIA.get(loteria)
    if limites is None:
        return n_jogos, n_dezenas, avisos

    min_dez, max_dez = limites

    if n_dezenas is None:
        return n_jogos, None, avisos

    original = int(n_dezenas)
    ajustado = original

    if ajustado < min_dez:
        ajustado = min_dez
    if ajustado > max_dez:
        ajustado = max_dez

    if ajustado != original:
        if min_dez == max_dez:
            avisos.append(
                f"Para {loteria}, dezenas/jogo é fixo em {min_dez}. "
                f"Valor ajustado de {original} para {ajustado}."
            )
        else:
            avisos.append(
                f"Para {loteria}, dezenas/jogo permitido: {min_dez}..{max_dez}. "
                f"Valor ajustado de {original} para {ajustado}."
            )

    return n_jogos, ajustado, avisos


# ---------- Thread de predição (para não travar o front) ----------
class PreditorThread(QThread):
    progresso = pyqtSignal(int)
    log = pyqtSignal(str)
    resultado = pyqtSignal(dict)

    def __init__(
        self,
        loteria: str,
        arquivo_csv: Path,
        n_jogos: int = 1,
        n_dezenas: int | None = None,
    ):
        super().__init__()
        self.loteria = loteria
        self.arquivo_csv = arquivo_csv
        self.n_jogos = n_jogos
        self.n_dezenas = n_dezenas

    def run(self) -> None:
        try:
            self.log.emit("🔍 Iniciando predição...")
            self.progresso.emit(0)

            df = carregar_dados(self.arquivo_csv)
            self.progresso.emit(3)
            self.log.emit("Dados carregados. Preparando calculos...")

            def _reportar_progresso(valor: int, mensagem: str = "") -> None:
                self.progresso.emit(int(valor))
                if mensagem:
                    self.log.emit(mensagem)

            palpites = gerar_palpite(
                df,
                self.loteria,
                n_jogos=self.n_jogos,
                n_dezenas=self.n_dezenas,
                progress_callback=_reportar_progresso,
            )

            self.progresso.emit(100)
            self.resultado.emit(palpites)
        except Exception as e:
            self.log.emit(f"❌ Erro na predição: {e}")


# ---------- Interface principal ----------
class App(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Previsor de Loterias Inteligente")
        self.setWindowIcon(QIcon("loteria.ico"))

        self.setMinimumSize(980, 560)
        self.resize(1080, 620)

        self._last_default_dezenas: int | None = None
        self._ultimo_resultado: dict | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        root = QVBoxLayout()
        root.setSpacing(10)
        root.setContentsMargins(12, 12, 12, 12)

        lbl_title = QLabel("🎯 Previsor de Loterias Inteligente")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_title.setObjectName("Title")
        root.addWidget(lbl_title)

        root.addWidget(self._build_config_box())
        root.addLayout(self._build_action_row())

        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress.setFormat("%p%")
        self.progress.setFixedHeight(26)
        root.addWidget(self.progress)

        self.alerta_dados = QLabel("")
        self.alerta_dados.setObjectName("AlertBanner")
        self.alerta_dados.setVisible(False)
        self.alerta_dados.setWordWrap(True)
        root.addWidget(self.alerta_dados)

        splitter = self._build_bottom_splitter()
        root.addWidget(splitter)

        self.setLayout(root)
        self._apply_style()

        self._sync_defaults_and_limits()
        self.combo_loteria.currentTextChanged.connect(self._sync_defaults_and_limits)

    # ---------- UI blocks ----------
    def _build_config_box(self) -> QGroupBox:
        box = QGroupBox("Configuração")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        grid.setContentsMargins(12, 14, 12, 12)

        lbl_loteria = QLabel("Loteria:")
        self.combo_loteria = QComboBox()
        self.combo_loteria.addItems(LOTERIAS)
        self.combo_loteria.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )

        lbl_concursos = QLabel("Concursos:")
        self.input_qtd = QLineEdit("10")
        self.input_qtd.setMaxLength(4)
        self.input_qtd.setPlaceholderText("Ex.: 10")

        lbl_n_jogos = QLabel("Jogos (saída):")
        self.input_n_jogos = QLineEdit("1")
        self.input_n_jogos.setMaxLength(3)
        self.input_n_jogos.setPlaceholderText("Ex.: 5")

        self.lbl_n_dezenas = QLabel("Dezenas/jogo:")
        self.input_n_dezenas = QLineEdit("")
        self.input_n_dezenas.setMaxLength(3)
        self.input_n_dezenas.setPlaceholderText("Ex.: 15")

        # Pequena melhoria: Enter nos campos dispara ação (opcional e sem quebrar)
        self.input_qtd.returnPressed.connect(self.iniciar_coleta)
        self.input_n_jogos.returnPressed.connect(self.predizer_jogo)
        self.input_n_dezenas.returnPressed.connect(self.predizer_jogo)

        grid.addWidget(lbl_loteria, 0, 0)
        grid.addWidget(self.combo_loteria, 0, 1)
        grid.addWidget(lbl_concursos, 0, 2)
        grid.addWidget(self.input_qtd, 0, 3)

        grid.addWidget(lbl_n_jogos, 1, 0)
        grid.addWidget(self.input_n_jogos, 1, 1)
        grid.addWidget(self.lbl_n_dezenas, 1, 2)
        grid.addWidget(self.input_n_dezenas, 1, 3)

        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(3, 1)

        box.setLayout(grid)
        return box

    def _build_action_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)

        self.btn_coletar = QPushButton("📥 Coletar Resultados")
        self.btn_coletar.clicked.connect(self.iniciar_coleta)

        self.btn_predizer = QPushButton("🧠 Prever Jogo Ideal")
        self.btn_predizer.clicked.connect(self.predizer_jogo)

        self.btn_salvar_relatorio = QPushButton("Salvar Relatorio")
        self.btn_salvar_relatorio.clicked.connect(self.salvar_relatorio)
        self.btn_salvar_relatorio.setDisabled(True)

        # Pequena melhoria: botão limpar log
        self.btn_limpar_log = QPushButton("🧹 Limpar Log")
        self.btn_limpar_log.clicked.connect(self._limpar_log)

        self.btn_coletar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.btn_predizer.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.btn_salvar_relatorio.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.btn_limpar_log.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )
        self.btn_salvar_relatorio.setFixedWidth(150)
        self.btn_limpar_log.setFixedWidth(140)

        row.addWidget(self.btn_coletar)
        row.addWidget(self.btn_predizer)
        row.addWidget(self.btn_salvar_relatorio)
        row.addWidget(self.btn_limpar_log)
        return row

    def _build_bottom_splitter(self) -> QSplitter:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        log_box = QGroupBox("📄 Log de Execução")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(10, 12, 10, 10)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Mensagens e status aparecerão aqui...")
        log_layout.addWidget(self.log_area)

        log_box.setLayout(log_layout)

        perf_box = QGroupBox("📊 Desempenho dos Modelos (F1/Acc médios)")
        perf_box.setTitle("Score dos Modelos (validacao temporal)")
        perf_layout = QVBoxLayout()
        perf_layout.setContentsMargins(10, 12, 10, 10)

        self.table_perf = QTableWidget(0, 4)
        self.table_perf.setHorizontalHeaderLabels(
            ["Modelo", "Histórico (%)", "Atual (%)", "Evolução (p.p. / %)"]
        )
        self.table_perf.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table_perf.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table_perf.verticalHeader().setVisible(False)
        self.table_perf.setAlternatingRowColors(True)

        perf_layout.addWidget(self.table_perf)
        perf_box.setLayout(perf_layout)

        splitter.addWidget(log_box)
        splitter.addWidget(perf_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        return splitter

    def _apply_style(self) -> None:
        # Pequenas melhorias visuais: foco, seleção, scrollbar e bordas mais suaves.
        self.setStyleSheet(
            """
            QWidget {
                background: #151515;
                color: #EAEAEA;
                font-size: 13px;
            }

            QLabel#Title {
                font-size: 19px;
                font-weight: 800;
                color: #FFFFFF;
                padding: 6px 0;
            }

            QLabel#AlertBanner {
                border: 1px solid #6A5420;
                border-radius: 10px;
                background: #2A2211;
                color: #F7E6B5;
                padding: 10px 12px;
                font-weight: 700;
            }

            QGroupBox {
                border: 1px solid #2E2E2E;
                border-radius: 12px;
                margin-top: 8px;
                padding: 10px;
                background: #1B1B1B;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #FFFFFF;
                font-weight: 800;
            }

            QLineEdit, QComboBox {
                height: 30px;
                border-radius: 9px;
                border: 1px solid #2C2C2C;
                background: #101010;
                padding: 0 10px;
                color: #FFFFFF;
                selection-background-color: #444444;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border: 1px solid #4A4A4A;
            }

            QComboBox::drop-down {
                border: 0px;
                width: 26px;
            }

            QPushButton {
                height: 36px;
                border-radius: 11px;
                border: 1px solid #2C2C2C;
                background: #222222;
                font-weight: 800;
            }
            QPushButton:hover { background: #2A2A2A; }
            QPushButton:pressed { background: #1F1F1F; }
            QPushButton:disabled {
                color: #888888;
                background: #1B1B1B;
                border: 1px solid #262626;
            }

            QProgressBar {
                border: 1px solid #2C2C2C;
                border-radius: 11px;
                text-align: center;
                background: #101010;
            }
            QProgressBar::chunk {
                border-radius: 11px;
                background: #3A3A3A;
            }

            QTextEdit {
                border-radius: 12px;
                border: 1px solid #2C2C2C;
                background: #0B0B0B;
                color: #FFFFFF;
                font-family: Consolas;
                font-size: 13px;
                padding: 8px;
            }

            QTableWidget {
                border-radius: 12px;
                border: 1px solid #2C2C2C;
                background: #111111;
                color: #FFFFFF;
                gridline-color: #2A2A2A;
                selection-background-color: #2A2A2A;
                selection-color: #FFFFFF;
            }
            QHeaderView::section {
                background-color: #222222;
                color: #FFFFFF;
                font-weight: 800;
                border: 0px;
                padding: 7px;
            }
            """
        )

    def _sync_defaults_and_limits(self) -> None:
        loteria = self.combo_loteria.currentText()
        default_dezenas = DEFAULT_DEZENAS_POR_LOTERIA.get(loteria, 6)
        min_dez, max_dez = LIMITES_DEZENAS_POR_LOTERIA.get(loteria, (1, 60))

        if min_dez == max_dez:
            self.lbl_n_dezenas.setText(f"Dezenas/jogo (fixo {min_dez}):")
        else:
            self.lbl_n_dezenas.setText(f"Dezenas/jogo ({min_dez}..{max_dez}):")

        tooltip = (
            f"{loteria}: permitido {min_dez}..{max_dez} dezenas por jogo."
            if min_dez != max_dez
            else f"{loteria}: quantidade fixa de {min_dez} dezenas por jogo."
        )
        if "lotomania" in loteria.lower():
            tooltip += "\nObs.: dezenas exibidas como 00..99 (inclui 00)."
        if _is_super_sete(loteria):
            tooltip += "\nObs.: Super Sete = 7 dígitos (0..9) por posição, pode repetir."

        self.input_n_dezenas.setToolTip(tooltip)

        atual = self.input_n_dezenas.text().strip()
        if not atual or (
            self._last_default_dezenas is not None
            and atual.isdigit()
            and int(atual) == self._last_default_dezenas
        ):
            self.input_n_dezenas.setText(str(default_dezenas))

        self._last_default_dezenas = int(default_dezenas)

    # ---------- coleta ----------
    def iniciar_coleta(self) -> None:
        loteria = self.combo_loteria.currentText()
        qtd_txt = self.input_qtd.text().strip()

        if not qtd_txt.isdigit() or int(qtd_txt) <= 0:
            QMessageBox.critical(self, "Erro", "Quantidade inválida.")
            return

        self._set_busy(True)
        self.log_area.clear()
        self.progress.setValue(0)
        self._append_log("🚀 Iniciando coleta...")

        self.thread = ColetorThread(loteria_nome=loteria, quantidade=int(qtd_txt))
        self.thread.progresso.connect(self.progress.setValue)
        self.thread.log.connect(self._append_log)
        self.thread.finalizado.connect(self._finalizar_coleta)
        self.thread.start()

    def _finalizar_coleta(self, df: pd.DataFrame) -> None:
        self._set_busy(False)

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

        n_jogos_txt = self.input_n_jogos.text().strip()
        n_dezenas_txt = self.input_n_dezenas.text().strip()

        if not n_jogos_txt.isdigit() or int(n_jogos_txt) <= 0:
            QMessageBox.critical(self, "Erro", "Quantidade de jogos inválida.")
            return
        n_jogos = int(n_jogos_txt)

        if n_dezenas_txt == "":
            n_dezenas = None
        else:
            if not n_dezenas_txt.isdigit() or int(n_dezenas_txt) <= 0:
                QMessageBox.critical(self, "Erro", "Dezenas por jogo inválido.")
                return
            n_dezenas = int(n_dezenas_txt)

        n_jogos, n_dezenas, avisos_ui = _ajustar_entrada_aposta(loteria, n_jogos, n_dezenas)

        if avisos_ui:
            for a in avisos_ui:
                self._append_log(f"⚠️ {a}")
            QMessageBox.warning(self, "Ajustes aplicados", "\n".join(avisos_ui))
            self.input_n_jogos.setText(str(n_jogos))
            if n_dezenas is not None:
                self.input_n_dezenas.setText(str(n_dezenas))

        self._set_busy(True)
        self._append_log("🧩 Preparando predição em segundo plano...")
        self.progress.setValue(0)

        self.pred_thread = PreditorThread(
            loteria=loteria,
            arquivo_csv=arquivo_csv,
            n_jogos=n_jogos,
            n_dezenas=n_dezenas,
        )
        self.pred_thread.log.connect(self._append_log)
        self.pred_thread.progresso.connect(self.progress.setValue)
        self.pred_thread.resultado.connect(self._exibir_resultado)
        self.pred_thread.finished.connect(lambda: self._set_busy(False))
        self.pred_thread.start()

    # ---------- exibição ----------
    def _exibir_resultado(self, palpites: dict) -> None:
        self._ultimo_resultado = dict(palpites)
        self.btn_salvar_relatorio.setDisabled(False)
        self._atualizar_alerta_dados(palpites.get("alertas_dados", []))
        self._append_log("✅ Predição concluída!\n")

        avisos = palpites.get("avisos", [])
        if isinstance(avisos, list) and avisos:
            for a in avisos:
                self._append_log(f"⚠️ {a}")
            QMessageBox.warning(self, "Ajustes/Restrições", "\n".join(map(str, avisos)))

        params = palpites.get("parametros", {})
        if isinstance(params, dict) and params:
            usada = params.get("n_dezenas_usada")
            solicitada = params.get("n_dezenas_solicitada")
            min_dez = params.get("min_dezenas")
            max_dez = params.get("max_dezenas")

            if usada is not None:
                self._append_log(
                    f"ℹ️ Parâmetros: dezenas/jogo usada={usada} "
                    f"(solicitada={solicitada}) | limites={min_dez}..{max_dez}"
                )
                try:
                    self.input_n_dezenas.setText(str(int(usada)))
                except Exception:
                    pass

        metodo_avaliacao = palpites.get("avaliacao_metodo")
        if isinstance(metodo_avaliacao, str) and metodo_avaliacao.strip():
            self._append_log(f"ðŸ“ AvaliaÃ§Ã£o: {metodo_avaliacao}")

        alertas_dados = palpites.get("alertas_dados", [])
        if isinstance(alertas_dados, list) and alertas_dados:
            for alerta in alertas_dados:
                if isinstance(alerta, dict):
                    self._append_log(
                        f"Alerta de dados: {alerta.get('titulo', 'Alerta')} - {alerta.get('mensagem', '')}"
                    )

        resumo_modelos = palpites.get("resumo_modelos", {})
        if isinstance(resumo_modelos, dict) and resumo_modelos.get("melhor_modelo"):
            self._append_log(
                f"ðŸ§­ Resumo: melhor modelo={resumo_modelos.get('melhor_modelo')} "
                f"| score={float(resumo_modelos.get('score_melhor_modelo', 0.0)) * 100.0:.2f}% "
                f"| confianca relativa={float(resumo_modelos.get('confianca_relativa', 0.0)) * 100.0:.2f}%"
            )

        pesos_modelos = palpites.get("pesos_modelos", {})
        if isinstance(pesos_modelos, dict) and pesos_modelos:
            pesos_txt = ", ".join(
                f"{nome}={float(valor):.2f}" for nome, valor in pesos_modelos.items()
            )
            self._append_log(f"âš–ï¸ Pesos do ensemble: {pesos_txt}")

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
                self._append_log(
                    f"🎯 Melhor combinação: {_format_dezenas(self.combo_loteria.currentText(), nums)}\n"
                )
            else:
                self._append_log(
                    f"• {chave}: {_format_dezenas(self.combo_loteria.currentText(), nums)}"
                )

        jogos = palpites.get("jogos_sugeridos", [])
        if jogos:
            self._append_log("\n🎟️ Jogos sugeridos:")
            for i, jogo in enumerate(jogos, start=1):
                self._append_log(
                    f"  Jogo {i}: {_format_jogo(self.combo_loteria.currentText(), jogo)}"
                )
            self._append_log("")

        self._atualizar_tabela_desempenho(palpites)

        if jogos:
            texto = "\n".join(
                [
                    f"Jogo {i}: {_format_jogo(self.combo_loteria.currentText(), j)}"
                    for i, j in enumerate(jogos, 1)
                ]
            )
            QMessageBox.information(self, "Previsão Concluída", f"Jogos sugeridos:\n{texto}")
            return

        melhor = palpites.get("melhor_combinacao", [])
        if melhor:
            QMessageBox.information(
                self,
                "Previsão Concluída",
                f"Melhor jogo previsto:\n{_format_dezenas(self.combo_loteria.currentText(), melhor)}",
            )

    def _atualizar_tabela_desempenho(self, palpites: dict) -> None:
        historico = _carregar_historico()
        atual = palpites.get("avaliacao_modelos", {})

        self.table_perf.setRowCount(0)

        if not atual:
            self._append_log("ℹ️ Nenhuma métrica de desempenho retornada pelos modelos.")
            return

        melhor_modelo = None
        melhor_score_atual = -1.0
        melhor_hist = 0.0
        melhor_delta_pct_rel = 0.0

        for modelo in sorted(atual.keys()):
            score_atual = float(atual.get(modelo, 0.0))
            score_hist = float(historico.get(modelo, 0.0))

            hist_pct = score_hist * 100.0
            atual_pct = score_atual * 100.0
            delta_pp = atual_pct - hist_pct

            if hist_pct > 0:
                delta_pct_rel = (delta_pp / hist_pct) * 100.0
            else:
                delta_pct_rel = 0.0

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
            item_evol = QTableWidgetItem(f"{emoji} {delta_pp:+.2f} p.p. / {delta_pct_rel:+.2f}%")

            for item in (item_modelo, item_hist, item_atual, item_evol):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_evol.setToolTip(
                f"Histórico: {hist_pct:.2f}%\n"
                f"Atual: {atual_pct:.2f}%\n"
                f"Variação: {delta_pp:+.2f} p.p. ({delta_pct_rel:+.2f}%)"
            )

            self.table_perf.setItem(linha, 0, item_modelo)
            self.table_perf.setItem(linha, 1, item_hist)
            self.table_perf.setItem(linha, 2, item_atual)
            self.table_perf.setItem(linha, 3, item_evol)

            if score_atual > melhor_score_atual:
                melhor_score_atual = score_atual
                melhor_modelo = modelo
                melhor_hist = score_hist
                melhor_delta_pct_rel = delta_pct_rel

        if melhor_modelo is not None:
            hist_pct = melhor_hist * 100.0
            atual_pct = melhor_score_atual * 100.0
            self._append_log(
                f"🏅 Melhor modelo no momento: {melhor_modelo} "
                f"→ Atual: {atual_pct:.2f}% | Histórico: {hist_pct:.2f}% "
                f"| Evolução relativa: {melhor_delta_pct_rel:+.2f}%"
            )

    def _atualizar_tabela_desempenho(self, palpites: dict) -> None:
        historico = palpites.get("historico_modelos", {})
        if not isinstance(historico, dict) or not historico:
            historico = _carregar_historico(self.combo_loteria.currentText())

        atual = palpites.get("avaliacao_modelos", {})
        detalhado = palpites.get("avaliacao_detalhada", {})

        self.table_perf.setRowCount(0)

        if not atual:
            self._append_log("Sem metricas de desempenho disponiveis.")
            return

        rotulos_metricas = {
            "score_composto": "Score composto",
            "taxa_acerto": "Taxa de acerto",
            "precisao_aposta": "Precisao da aposta",
            "jaccard": "Jaccard",
            "f1_micro": "F1 micro",
            "acuracia_exata": "Acuracia exata",
            "f1_ponderado": "F1 ponderado",
            "acerto_4_mais": "Acerto 4+",
            "acerto_exato": "Acerto exato",
            "acertos_medios": "Acertos medios",
            "amostras_avaliadas": "Amostras",
        }
        metricas_percentuais = {
            "score_composto",
            "taxa_acerto",
            "precisao_aposta",
            "jaccard",
            "f1_micro",
            "acuracia_exata",
            "f1_ponderado",
            "acerto_4_mais",
            "acerto_exato",
        }

        melhor_modelo = None
        melhor_score_atual = -1.0
        melhor_hist = 0.0
        melhor_delta_pct_rel = 0.0

        for modelo in atual.keys():
            score_atual = float(atual.get(modelo, 0.0))
            score_hist = float(historico.get(modelo, 0.0))

            hist_pct = score_hist * 100.0
            atual_pct = score_atual * 100.0
            delta_pp = atual_pct - hist_pct
            delta_pct_rel = (delta_pp / hist_pct) * 100.0 if hist_pct > 0 else 0.0

            if score_hist == 0 and score_atual > 0:
                emoji = "NEW"
            elif delta_pp > 0:
                emoji = "UP"
            elif delta_pp < 0:
                emoji = "DOWN"
            else:
                emoji = "EQ"

            detalhes_modelo = detalhado.get(modelo, {})
            tooltip_linhas = [
                f"Historico: {hist_pct:.2f}%",
                f"Atual: {atual_pct:.2f}%",
                f"Variacao: {delta_pp:+.2f} p.p. ({delta_pct_rel:+.2f}%)",
            ]
            if isinstance(detalhes_modelo, dict):
                for chave, valor in detalhes_modelo.items():
                    rotulo = rotulos_metricas.get(chave, chave.replace("_", " ").title())
                    if chave in metricas_percentuais:
                        tooltip_linhas.append(f"{rotulo}: {float(valor) * 100.0:.2f}%")
                    elif chave == "amostras_avaliadas":
                        tooltip_linhas.append(f"{rotulo}: {int(float(valor))}")
                    else:
                        tooltip_linhas.append(f"{rotulo}: {float(valor):.2f}")
            tooltip = "\n".join(tooltip_linhas)

            linha = self.table_perf.rowCount()
            self.table_perf.insertRow(linha)

            item_modelo = QTableWidgetItem(modelo)
            item_hist = QTableWidgetItem(f"{hist_pct:.2f}%")
            item_atual = QTableWidgetItem(f"{atual_pct:.2f}%")
            item_evol = QTableWidgetItem(f"{emoji} {delta_pp:+.2f} p.p. / {delta_pct_rel:+.2f}%")

            for item in (item_modelo, item_hist, item_atual, item_evol):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setToolTip(tooltip)

            self.table_perf.setItem(linha, 0, item_modelo)
            self.table_perf.setItem(linha, 1, item_hist)
            self.table_perf.setItem(linha, 2, item_atual)
            self.table_perf.setItem(linha, 3, item_evol)

            if score_atual > melhor_score_atual:
                melhor_score_atual = score_atual
                melhor_modelo = modelo
                melhor_hist = score_hist
                melhor_delta_pct_rel = delta_pct_rel

        if melhor_modelo is not None:
            hist_pct = melhor_hist * 100.0
            atual_pct = melhor_score_atual * 100.0
            self._append_log(
                f"Melhor modelo no momento: {melhor_modelo} "
                f"-> Score atual: {atual_pct:.2f}% | Historico: {hist_pct:.2f}% "
                f"| Evolucao relativa: {melhor_delta_pct_rel:+.2f}%"
            )

    # ---------- helpers ----------
    def _append_log(self, msg: str) -> None:
        self.log_area.append(msg)
        self.log_area.moveCursor(QTextCursor.MoveOperation.End)

    def _limpar_log(self) -> None:
        self.log_area.clear()
        self._append_log("🧹 Log limpo.")

    def _atualizar_alerta_dados(self, alertas: list) -> None:
        if not isinstance(alertas, list) or not alertas:
            self.alerta_dados.clear()
            self.alerta_dados.setVisible(False)
            return

        alertas_validos = [a for a in alertas if isinstance(a, dict)]
        if not alertas_validos:
            self.alerta_dados.clear()
            self.alerta_dados.setVisible(False)
            return

        prioridade = {"warning": 2, "info": 1}
        principal = sorted(
            alertas_validos,
            key=lambda a: prioridade.get(str(a.get("nivel", "info")), 0),
            reverse=True,
        )[0]

        nivel = str(principal.get("nivel", "info"))
        titulo = str(principal.get("titulo", "Alerta"))
        mensagem = str(principal.get("mensagem", ""))
        self.alerta_dados.setText(f"{titulo}: {mensagem}")
        self.alerta_dados.setVisible(True)

        if nivel == "warning":
            self.alerta_dados.setStyleSheet(
                "border: 1px solid #8A5A12; border-radius: 10px; background: #33240F; "
                "color: #FFD889; padding: 10px 12px; font-weight: 700;"
            )
        else:
            self.alerta_dados.setStyleSheet(
                "border: 1px solid #33556E; border-radius: 10px; background: #162633; "
                "color: #B9E4FF; padding: 10px 12px; font-weight: 700;"
            )

    def salvar_relatorio(self) -> None:
        if not isinstance(self._ultimo_resultado, dict) or not self._ultimo_resultado:
            QMessageBox.information(self, "Relatorio", "Nenhuma previsao para salvar ainda.")
            return

        loteria = self.combo_loteria.currentText().replace(" ", "_").replace("+", "mais")
        caminho_sugerido = str(Path.cwd() / f"{loteria}_ultimo_relatorio.txt")
        caminho, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar relatorio",
            caminho_sugerido,
            "Arquivos de texto (*.txt)",
        )
        if not caminho:
            return

        linhas: List[str] = [f"Loteria: {self.combo_loteria.currentText()}"]
        params = self._ultimo_resultado.get("parametros", {})
        if isinstance(params, dict):
            linhas.append(f"Jogos gerados: {params.get('n_jogos', '-')}")
            linhas.append(f"Dezenas usadas: {params.get('n_dezenas_usada', '-')}")

        metodo = self._ultimo_resultado.get("avaliacao_metodo")
        if isinstance(metodo, str) and metodo.strip():
            linhas.extend(["", f"Avaliacao: {metodo}"])

        for chave in [
            "frequencia_simples",
            "recencia_ponderada",
            "random_forest",
            "logistic_regression",
            "k_nearest_neighbors",
            "gradient_boosting",
            "melhor_combinacao",
        ]:
            nums = self._ultimo_resultado.get(chave, [])
            if nums:
                linhas.append(
                    f"{chave}: {_format_dezenas(self.combo_loteria.currentText(), nums)}"
                )

        jogos = self._ultimo_resultado.get("jogos_sugeridos", [])
        if jogos:
            linhas.extend(["", "Jogos sugeridos:"])
            for i, jogo in enumerate(jogos, start=1):
                linhas.append(
                    f"Jogo {i}: {_format_jogo(self.combo_loteria.currentText(), jogo)}"
                )

        avaliacao = self._ultimo_resultado.get("avaliacao_modelos", {})
        if isinstance(avaliacao, dict) and avaliacao:
            linhas.extend(["", "Score dos modelos:"])
            for nome, score in avaliacao.items():
                linhas.append(f"{nome}: {float(score) * 100.0:.2f}%")

        Path(caminho).write_text("\n".join(linhas), encoding="utf-8")
        self._append_log(f"Relatorio salvo em: {Path(caminho).name}")
        QMessageBox.information(self, "Relatorio", "Relatorio salvo com sucesso.")

    def _set_busy(self, busy: bool) -> None:
        # Pequena melhoria: bloqueia ações enquanto roda thread
        self.btn_coletar.setDisabled(busy)
        self.btn_predizer.setDisabled(busy)
        self.combo_loteria.setDisabled(busy)
        self.input_qtd.setDisabled(busy)
        self.input_n_jogos.setDisabled(busy)
        self.input_n_dezenas.setDisabled(busy)
        if busy:
            self.alerta_dados.setVisible(False)
        if busy:
            self.btn_salvar_relatorio.setDisabled(True)
        elif isinstance(self._ultimo_resultado, dict) and self._ultimo_resultado:
            self.btn_salvar_relatorio.setDisabled(False)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    janela = App()
    janela.show()
    sys.exit(app.exec())
