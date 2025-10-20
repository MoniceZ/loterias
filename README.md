# 🎯 Coletor e Preditor Inteligente de Resultados de Loterias

Aplicação completa em **Python + PyQt6** para coleta automática de resultados das principais loterias brasileiras e geração de previsões inteligentes com base em **modelos estatísticos** e **aprendizado de máquina**.

Agora com:
- Interface moderna e responsiva (PyQt6)
- Execução em threads (sem travamentos)
- Histórico automático de desempenho dos modelos
- Comparativo direto de desempenho no front-end
- Aprendizado adaptativo: o sistema melhora com o tempo

---

## 🚀 Funcionalidades

- **Coleta automática** dos concursos direto do site [asloterias.com.br](https://asloterias.com.br)
- **Armazenamento local em CSV** para histórico de resultados
- **Predição com 6 abordagens distintas:**
  - Frequência simples  
  - Recência ponderada  
  - Random Forest  
  - Regressão Logística  
  - K-Nearest Neighbors (KNN)  
  - Gradient Boosting
- **Combinação final inteligente (ensemble)** com pesos dinâmicos baseados em desempenho real
- **Avaliação contínua dos modelos (F1-score e acurácia média)**
- **Histórico salvo automaticamente** em `historico_modelos.csv`
- **Interface aprimorada em PyQt6** com:
  - Barra de progresso em tempo real  
  - Log com emojis e cursor automático  
  - Tabela de desempenho dos modelos  
  - Indicadores 🔼🔽 de melhora ou queda comparado ao histórico anterior  

---

## 🧠 Aprendizado Adaptativo

O sistema **aprende e se calibra automaticamente**:
1. Cada execução registra o desempenho dos modelos no arquivo `historico_modelos.csv`.  
2. Em novas previsões, o sistema **ajusta os pesos** de cada modelo com base nas médias históricas.  
3. Modelos mais precisos recebem mais peso na votação final.  
4. O front mostra se o desempenho **melhorou (🔼)** ou **piorou (🔽)** em relação à execução anterior.

---

## 🧩 Tecnologias Utilizadas

| Categoria | Tecnologias |
|------------|--------------|
| Linguagem | Python 3.10+ |
| Interface Gráfica | PyQt6 |
| Coleta de Dados | Selenium WebDriver (Chrome) + webdriver-manager |
| Ciência de Dados | pandas, numpy |
| Aprendizado de Máquina | scikit-learn |
| Persistência | CSV local (UTF-8) |
| Visual | QProgressBar, QTextEdit, QTableWidget |

---

## 🗂️ Estrutura do Projeto

```text
projeto_loteria/
├── main.py             # Ponto de entrada da aplicação (executa o app)
├── ui.py               # Interface gráfica (PyQt6)
├── coleta.py           # Módulo de coleta automática via Selenium
├── predicao.py         # Modelos estatísticos e de ML com aprendizado adaptativo
├── historico_modelos.csv  # Histórico de desempenho (gerado automaticamente)
├── requirements.txt    # Dependências do projeto
└── loteria.ico         # Ícone do programa (opcional)
