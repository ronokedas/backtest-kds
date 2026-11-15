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

## 🔬 Laboratório de Estratégias Desktop Binance (`strategies/desktop_binance/`)

O repositório inclui as **10 estratégias táticas em Pine Script / Texto** da pasta `strategies/desktop_binance/`, acompanhadas da **Directiva Institucional para IA (`INSTRUCOES_IA_QUANT_RESEARCH.md`)** e do testador quantitativo automatizado (`test_desktop_strategies.py`).

### As 10 Estratégias Disponíveis:
1. `01_padrao_123_stormer.txt`: Pivô de reversão com filtro de tendência na EMA 100.
2. `02_quatro_emas_alinhadas.txt`: Continuação de tendência em 4 EMAs (9, 21, 50, 200).
3. `03_cci_rsi_40_80.txt`: Momentum e cruzamento simultâneo de osciladores.
4. `04_choch_fibo_618.txt`: Quebra de estrutura (CHoCH) + Retração na Fibo 61.8% + Engolfo.
5. `05_didi_index_agulhada.txt`: Agulhada do Didi Index clássica com médias 3, 8 e 20.
6. `06_divergencia_rsi_price.txt`: Divergência clássica entre topos/fundos de preço e RSI.
7. `07_euforia_rejeicao_volume.txt`: Clímax de volume (> 2.2x média) com pavio de rejeição > 55%.
8. `08_smart_money_sweep_liquidity.txt`: Varredura de liquidez de 20 períodos com retorno ao range.
9. `09_mfi_rsi_extremo_reversao.txt`: Exaustão dupla em zonas extremas de sobrevenda/sobrecompra.
10. `10_mola_bollinger_reversao.txt`: Efeito mola nas Bandas de Bollinger com pavio de rejeição.

### Como a IA / Usuário Testa as Estratégias:
```bash
# Testar uma estratégia no modo puro (baseline de varejo):
python scripts/test_desktop_strategies.py --strategy 08_smart_money_sweep_liquidity

# Testar a MESMA estratégia com Confluências Institucionais (CVD + Volume + Exaustão):
python scripts/test_desktop_strategies.py --strategy 08_smart_money_sweep_liquidity --confluence

# Executar o Benchmark Comparativo de TODAS as 10 estratégias nos 30 pares:
python scripts/test_desktop_strategies.py --benchmark
```

> 📖 **Instruções Detalhadas para IA:** Consulte o arquivo [`strategies/desktop_binance/INSTRUCOES_IA_QUANT_RESEARCH.md`](strategies/desktop_binance/INSTRUCOES_IA_QUANT_RESEARCH.md) para o guia completo de pesquisa quantitativa anti-repaint e busca de alfa.

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
├── strategies/                # Repositório de Estratégias e Modelos
│   └── desktop_binance/       # 10 Estratégias Pine Script + Diretivas para IA
│       ├── INSTRUCOES_IA_QUANT_RESEARCH.md # Diretrizes completas para a IA
│       ├── 01_padrao_123_stormer.txt
│       ├── 04_choch_fibo_618.txt
│       ├── 08_smart_money_sweep_liquidity.txt
│       └── ... (10 estratégias)
├── scripts/                   # Scripts auxiliares e de pesquisa
│   ├── run_massive_backtest.py
│   ├── test_desktop_strategies.py # Testador quantitativo das estratégias
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

