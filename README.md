# QuantumFlow Backtest KDS 🚀

Sistema autônomo de **Backtest Quantitativo Institucional** e replay histórico anti-repaint desenvolvido para validação estatística de estratégias de fluxo de ordens e Smart Money Concepts (SMC) nos 30 principais pares da **Binance Futures (USDⓈ-M)**.

---

## 📊 Base de Dados Histórica (.CSV)

O repositório inclui a base completa de dados reais baixados diretamente da API da Binance Futures, cobrindo **90 arquivos `.csv`** na pasta `data/historical/`:

* **30 Pares no Timeframe M15 (`{PAR}_15m_1000.csv`)**: 1.000 velas de 15 minutos com dados de Taker Buy Volume, Taker Sell Volume, Delta e CVD acumulado.
* **30 Pares no Timeframe M5 (`{PAR}_5m_1000.csv`)**: 1.000 velas de 5 minutos com Volume Relativo (RVOL) e volatilidade para gatilhos táteis de engolfo.
* **30 Pares de Open Interest (`{PAR}_oi_15m_500.csv`)**: 500 períodos de contratos em aberto para detecção de *Short Trap* e *Long Trap*.

### Pares Inclusos (Top 30 Binance Futures):
> `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `DOGEUSDT`, `ADAUSDT`, `AVAXUSDT`, `LINKUSDT`, `SUIUSDT`, `NEARUSDT`, `APTUSDT`, `1000PEPEUSDT`, `FETUSDT`, `RENDERUSDT`, `TAOUSDT`, `ARBUSDT`, `OPUSDT`, `INJUSDT`, `TIAUSDT`, `SEIUSDT`, `WIFUSDT`, `AAVEUSDT`, `DOTUSDT`, `ATOMUSDT`, `POLUSDT`, `LTCUSDT`, `UNIUSDT`, `BCHUSDT`, `TRXUSDT`.

---

## 🧠 Confluências da Estratégia Institucional

O motor de replay histórico (`InstitutionalReplayBacktester`) simula a execução barra a barra com **fricção real (Taxa Taker 0.05% + Slippage 0.02%)** e **Zero Viés Futuro (No Look-Ahead Bias)**:

1. **Localização em Fibonacci:** O preço visita a *Golden Zone* (50% a 61.8%) da pernada anterior.
2. **Smart Money Structure:** Identificação estrita de quebra de caráter (**CHoCH**) e quebra contínua (**BOS**).
3. **CVD Retest Divergence:** Quando o preço volta para testar a região de 50%, o CVD (Cumulative Volume Delta) não confirma o rompimento, acusando exaustão ou absorção passiva.
4. **Volume Profile (70% Value Area):** Rejeição e sweep de liquidez em **VAH** (teto) ou **VAL** (piso), tendo o **POC** (Point of Control) como ímã de liquidez.
5. **Open Interest Trap Tracker:** Detecção de players presos em posições perdedoras forçando liquidações em cascata.
6. **Gatilho Sniper no M5:** Fechamento de vela de engolfo a favor da operação após a exaustão.
7. **Risco/Retorno Assimétrico:** Relação mínima de **2:1 (R:R 2.0)** com invalidação técnica ancorada na mínima/máxima do reteste.

---

## 💻 Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/ronokedas/backtest-kds.git
cd backtest-kds
pip install -r requirements.txt
```

---

## ⚡ Como Executar os Backtests

### 1. Executar Backtest em Todos os 30 Pares
Executa a bateria massiva consolidada calculando taxa de acerto global, Profit Factor e Expectativa Matemática:

```bash
python run.py
```

### 2. Executar em um Par Específico (ex: SOLUSDT ou BTCUSDT)
```bash
python run.py --symbol SOLUSDT
python run.py --symbol BTCUSDT
python run.py --symbol ETHUSDT
```

### 3. Ajustar Parâmetros de Filtro (Score de Confluência e R:R)
```bash
python run.py --symbol SOLUSDT --min-score 75.0 --rr 2.0
```

---

## 📁 Estrutura do Projeto

```
backtest-kds/
├── backtest/                  # Motores de Replay Histórico
│   ├── engine.py              # Motor padrão barra a barra
│   └── institutional_replay.py # Motor institucional com atrito real e BE
├── data/
│   ├── historical/            # 90 arquivos .CSV com histórico de 30 pares
│   ├── data_validator.py      # Validação e limpeza de candles
│   └── symbol_universe.py     # Definição dos universos de pares
├── indicators/                # Indicadores matemáticos e de fluxo
│   ├── cvd.py                 # Delta e CVD acumulado
│   ├── absorption.py          # Métricas de absorção passiva
│   ├── confluence.py          # Fibonacci e pontuação multidimensional
│   ├── market_structure.py    # CHoCH e BOS anti-repaint
│   ├── volume_profile.py      # POC, VAH, VAL e sweeps
│   └── oi_tracker.py          # Open Interest traps e desvios
├── scripts/                   # Scripts auxiliares e calibração
│   ├── run_massive_backtest.py
│   ├── evaluate_sniper_geometry.py
│   └── fast_calibrate.py
├── run.py                     # Entrypoint principal via linha de comando
├── config.yaml                # Parâmetros e pesos das confluências
├── requirements.txt           # Dependências do Python
└── README.md                  # Documentação completa
```

---

## ⚖️ Isenção de Responsabilidade (Disclaimer)

Este projeto foi desenvolvido para fins educacionais, de pesquisa quantitativa e uso pessoal em ambiente de demonstração/paper trading. Resultados passados não garantem lucros futuros no mercado de criptoativos e derivativos de alto risco.
