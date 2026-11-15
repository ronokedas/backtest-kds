"""QuantumFlow Backtest Runner — Executador Principal.

Executa o replay histórico institucional anti-repaint utilizando a base de dados
local em CSV (M15, M5 e Open Interest) de 30 pares da Binance Futures.

Uso:
    python run.py                     # Executa o backtest completo nos 30 pares
    python run.py --symbol SOLUSDT    # Executa apenas no par especificado
    python run.py --min-score 75.0    # Ajusta o corte de score institucional
    python run.py --rr 2.0            # Define a relação risco/retorno (padrão 2:1)
"""

import argparse
import sys
from pathlib import Path

# Adiciona o diretório raiz ao PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.run_massive_backtest import run_massive_backtest, TOP_30_PAIRS


def main():
    parser = argparse.ArgumentParser(description="QuantumFlow Backtest Suite — 30 Pares Binance Futures")
    parser.add_argument("--symbol", type=str, default=None, help="Par específico para testar (ex: BTCUSDT, SOLUSDT)")
    parser.add_argument("--min-score", type=float, default=70.0, help="Score mínimo de confluência (padrão: 70.0)")
    parser.add_argument("--rr", type=float, default=2.0, help="Relação Risco:Retorno (padrão: 2.0)")

    args = parser.parse_args()

    symbols = [args.symbol.upper()] if args.symbol else TOP_30_PAIRS

    print("\n" + "=" * 70)
    print("      🚀 INICIANDO QUANTUMFLOW BACKTEST REPLAY INSTITUCIONAL")
    print("=" * 70)
    print(f"Pares selecionados:   {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"Score de Confluência: >= {args.min_score}")
    print(f"Risco/Retorno (R:R):  {args.rr}:1")
    print("Origem dos dados:     data/historical/*.csv (Binance Futures)")
    print("=" * 70 + "\n")

    report = run_massive_backtest(symbols=symbols, min_score=args.min_score, risk_reward=args.rr)

    print("\n" + "=" * 70)
    print("              RESULTADO CONSOLIDADO DO BACKTEST")
    print("=" * 70)
    print(f"Pares Testados:       {report['symbols_tested']}")
    print(f"Total de Trades:      {report['total_trades']}")
    print(f"Trades Vencedores:    {report['wins']}")
    print(f"Trades Perdedores:    {report['losses']}")
    print(f"Taxa de Acerto (WR):  {report['global_win_rate_pct']}%")
    print(f"Profit Factor (PF):   {report['global_profit_factor']}")
    print(f"Relação R:R Mínima:   {report['risk_reward_ratio']}:1")
    print(f"Expectativa Matemática: {report['global_expectancy_r']:+.2f} R por trade")
    print(f"Resultado Líquido:    {report['global_net_pnl_r']:+.2f} R acumulados")
    print("=" * 70 + "\n")

    if args.symbol and args.symbol.upper() in report.get("by_symbol", {}):
        sym_res = report["by_symbol"][args.symbol.upper()]
        print(f"--- Detalhes para {args.symbol.upper()} ---")
        print(f"Trades: {sym_res.get('trades_count', 0)} | WR: {sym_res.get('win_rate_pct', 0)}% | PF: {sym_res.get('profit_factor', 0)} | PnL: {sym_res.get('net_pnl_r', 0):+.2f}R")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
