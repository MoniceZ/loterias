# Coletor e Preditor de Resultados de Loterias

Aplicação em Python com interface gráfica (PyQt6) para coleta automática de resultados de loterias (Mega Sena e Lotofácil) e geração de previsões baseadas em modelos estatísticos e de aprendizado de máquina.

## Funcionalidades

- Coleta automatizada de concursos direto do site [asloterias.com.br](https://asloterias.com.br)
- Armazenamento em CSV dos concursos coletados
- Geração de palpites com 6 estratégias:
  - Frequência simples
  - Recência ponderada
  - Random Forest
  - Regressão Logística
  - KNN
  - Gradient Boosting
- Combinação final por votação entre modelos

## Tecnologias Utilizadas

- Python 3.10+
- PyQt6
- Selenium WebDriver (Chrome)
- pandas e numpy
- scikit-learn
- webdriver-manager

## Estrutura do Projeto

```text
projeto_loteria/
├── main.py             # Entrada principal da aplicação
├── ui.py               # Interface gráfica
├── coleta.py           # Lógica de coleta dos concursos via Selenium
├── predicao.py         # Algoritmos de predição
├── requirements.txt    # Lista de dependências exatas
└── loteria.ico         # Ícone do programa (opcional)
```

## Requisitos

- Python 3.10 ou superior
- Google Chrome instalado
- Internet ativa
- Sistema operacional compatível com o ChromeDriver (Windows, Linux, Mac)

## Aviso Legal

Este sistema é apenas para fins educacionais e experimentais. Não há garantia de acerto nos resultados. Jogar loteria envolve risco financeiro. Use por sua conta e risco.

## Licença

MIT License — Uso livre para fins pessoais e comerciais.
