"""Executes real-data backtest on Binance USDⓈ-M Futures historical klines."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.binance_rest import BinanceRestClient
from backtest.engine import AbsorptionBacktester


async def main():
    print("=== EXECUTANDO BACKTEST COM DADOS REAIS DA BINANCE ===")
    client = BinanceRestClient()
    tester = AbsorptionBacktester(min_score=65.0, risk_reward_ratio=2.0)

    try:
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            print(f"\nBaixando histórico real de klines (15m) para {symbol}...")
            klines = await client.get_klines(symbol=symbol, interval="15m", limit=500)
            print(f"Total de candles carregados: {len(klines)}")

            results = tester.run_backtest(klines)
            print(f"--- Relatório de Performance ({symbol} - 15m) ---")
            print(f"Trades Disparados: {results.get('total_trades', 0)}")
            if results.get("total_trades", 0) > 0:
                print(f"Taxa de Acerto: {results['win_rate_pct']}%")
                print(f"Expectativa Líquida por Trade: {results['expectancy_r']:+.2f} R")
                print(f"PnL Total Acumulado: {results['total_pnl_r']:+.2f} R")
                print(f"Profit Factor: {results['profit_factor']:.2f}")
                print(f"Max Drawdown: {results['max_drawdown_pct']:.2f}%")
            else:
                print("Nenhum trade disparado no período com score >= 65.")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())

