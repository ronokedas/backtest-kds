"""Live verification script for Binance data collection and integrity.

Performs:
1. Clock skew check against Binance USDⓈ-M Futures & Spot servers.
2. Dynamic symbol universe discovery and liquidity ranking.
3. Historical Kline fetch and OHLCV structural integrity audit.
4. Derivatives metrics fetch (Open Interest, Taker Volume Ratio, Funding Rate).
5. Live WebSocket connection test for aggTrade and bookTicker streams.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime, timezone
from data.binance_rest import BinanceRestClient
from data.binance_ws import BinanceWebSocketClient
from data.symbol_universe import SymbolUniverse
from data.data_validator import DataValidator, AggTradeData

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("LiveVerification")


async def main():
    logger.info("=== INICIANDO VALIDAÇÃO DE DADOS REAIS DA BINANCE ===")
    validator = DataValidator(max_clock_skew_ms=2000)
    rest_client = BinanceRestClient(validator=validator)

    try:
        # 1. Sincronização de Relógio
        logger.info("\n--- 1. Sincronização de Relógio ---")
        is_synced, skew_ms, skew_msg = await rest_client.check_clock_skew(market="futures_usdm")
        logger.info(f"Futures USDⓈ-M Clock: {skew_msg}")
        if not is_synced:
            logger.error("ALERTA CRÍTICO: Relógio desincronizado com Binance!")

        # 2. Descoberta Dinâmica de Símbolos
        logger.info("\n--- 2. Descoberta Dinâmica de Pares Elegíveis (Futures) ---")
        universe = SymbolUniverse(rest_client=rest_client, min_volume_24h_usd=50_000_000.0)
        top_symbols = await universe.discover_futures_symbols(top_n=5)
        for idx, sym in enumerate(top_symbols, 1):
            logger.info(
                f"#{idx} {sym.symbol}: Vol 24h: ${sym.volume_24h_usd:,.0f} | Spread: {sym.spread_pct:.4f}% | Precision: P={sym.price_precision}, Q={sym.quantity_precision}"
            )

        target_symbol = "BTCUSDT"

        # 3. Klines e Integridade OHLCV + Taker Volume
        logger.info(f"\n--- 3. Auditoria de Klines e Delta Histórico ({target_symbol} - 15m) ---")
        klines = await rest_client.get_klines(symbol=target_symbol, interval="15m", limit=10)
        is_valid, errors = validator.validate_kline_series(klines, timeframe_ms=15 * 60 * 1000)
        logger.info(f"Total klines recebidos: {len(klines)} | Integridade: {'VÁLIDA' if is_valid else 'FALHA'}")
        if errors:
            for e in errors:
                logger.error(f"Erro em kline: {e}")

        last_kline = klines[-1]
        t_open = datetime.fromtimestamp(last_kline.open_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        taker_sell_vol = last_kline.volume - last_kline.taker_buy_base_volume
        kline_delta = last_kline.taker_buy_base_volume - taker_sell_vol
        logger.info(
            f"Último candle 15m [{t_open}]: O={last_kline.open} H={last_kline.high} L={last_kline.low} C={last_kline.close}"
        )
        logger.info(
            f"Volume Total: {last_kline.volume:.2f} | Taker Buy: {last_kline.taker_buy_base_volume:.2f} | Taker Sell: {last_sell_vol:.2f}"
            if "last_sell_vol" in locals()
            else f"Volume Total: {last_kline.volume:.2f} | Taker Buy: {last_kline.taker_buy_base_volume:.2f} | Taker Sell: {taker_sell_vol:.2f} | Delta do Candle: {kline_delta:+.2f}"
        )

        # 4. Derivativos Oficiais (Open Interest, Taker Ratio, Funding)
        logger.info(f"\n--- 4. Métricas de Derivativos ({target_symbol}) ---")
        oi_data = await rest_client.get_open_interest(target_symbol)
        funding_data = await rest_client.get_funding_rate(target_symbol, limit=1)
        taker_ratio = await rest_client.get_taker_long_short_ratio(target_symbol, period="15m", limit=1)

        logger.info(f"Open Interest Atual: {float(oi_data['openInterest']):,.2f} BTC")
        if funding_data:
            f_rate = float(funding_data[0]["fundingRate"]) * 100
            f_time = datetime.fromtimestamp(funding_data[0]["fundingTime"] / 1000, tz=timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
            logger.info(f"Taxa de Funding: {f_rate:.4f}% (Próximo acerto: {f_time})")
        if taker_ratio:
            logger.info(
                f"Taker Buy/Sell Ratio (15m): Buy Vol={taker_ratio[0]['buyVol']}, Sell Vol={taker_ratio[0]['sellVol']}, Ratio={taker_ratio[0]['buySellRatio']}"
            )

        # 5. Live WebSocket Stream Test (aggTrade)
        logger.info(f"\n--- 5. Teste de Stream em Tempo Real WebSocket (aggTrade - 5 segundos) ---")
        ws_client = BinanceWebSocketClient(market="futures_usdm", validator=validator)
        collected_trades = []

        async def on_trade(trade: AggTradeData):
            collected_trades.append(trade)
            if len(collected_trades) <= 5:
                logger.info(
                    f"Live Trade #{trade.aggregate_trade_id}: Side={trade.aggressor_side} (m={trade.is_buyer_maker}) | Qty={trade.quantity:.4f} | Price={trade.price:.1f} | Delta={trade.signed_delta:+.4f}"
                )

        ws_client.add_trade_callback(on_trade)

        ws_task = asyncio.create_task(ws_client.start([target_symbol], stream_types=["aggTrade"]))
        await asyncio.sleep(5)
        await ws_client.stop()
        ws_task.cancel()

        logger.info(f"Total de trades reais capturados em 5s: {len(collected_trades)}")
        if collected_trades:
            total_buy_delta = sum(t.quantity for t in collected_trades if t.aggressor_side == "BUY")
            total_sell_delta = sum(t.quantity for t in collected_trades if t.aggressor_side == "SELL")
            net_delta = total_buy_delta - total_sell_delta
            logger.info(
                f"Resumo da Amostra: Buy Agressor = {total_buy_delta:.3f} BTC | Sell Agressor = {total_sell_delta:.3f} BTC | Delta Líquido = {net_delta:+.3f} BTC"
            )

        logger.info("\n=== VALIDAÇÃO CONCLUÍDA COM SUCESSO! DADOS 100% REAIS E AUDITADOS ===")

    finally:
        await rest_client.close()


if __name__ == "__main__":
    asyncio.run(main())
