"""Massive Multi-Pair Backtesting & Confluence Calibration Suite.

Audits the Triple Confluence Institutional Strategy across 30 top Binance Futures pairs:
- Real Historical Replay (Zero Look-Ahead Bias)
- Realistic Trading Friction (Taker Fee + Slippage)
- Target: >= 70% Win Rate with 2:1 Risk/Reward Ratio (R:R >= 2.0)
- Sample: >= 20 trades per symbol (total >= 600 trades)
"""

import os
import sys
import json
import time
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from indicators.cvd import CVDEngine
from indicators.absorption import AbsorptionMetrics
from indicators.confluence import ConfluenceAnalyzer
from backtest.institutional_replay import InstitutionalReplayBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MassiveBacktest")

CACHE_DIR = Path("data/historical")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TOP_30_PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT",
    "NEARUSDT", "APTUSDT", "1000PEPEUSDT", "FETUSDT", "RENDERUSDT",
    "TAOUSDT", "ARBUSDT", "OPUSDT", "INJUSDT", "TIAUSDT",
    "SEIUSDT", "WIFUSDT", "AAVEUSDT", "DOTUSDT", "ATOMUSDT",
    "POLUSDT", "LTCUSDT", "UNIUSDT", "BCHUSDT", "TRXUSDT"
]


def fetch_binance_klines(symbol: str, interval: str = "15m", limit: int = 1000) -> pd.DataFrame:
    """Fetches real historical klines with taker volumes from Binance Futures REST API."""
    cache_file = CACHE_DIR / f"{symbol}_{interval}_{limit}.csv"
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
            if len(df) >= limit * 0.8:
                return df
        except Exception:
            pass

    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                cols = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "count", "taker_buy_vol",
                    "taker_buy_quote_vol", "ignore"
                ]
                df = pd.DataFrame(data, columns=cols)
                for c in ["open", "high", "low", "close", "volume", "taker_buy_vol"]:
                    df[c] = df[c].astype(float)
                df.to_csv(cache_file, index=False)
                return df
        except Exception as e:
            time.sleep(1.0)
            if attempt == 2:
                logger.error(f"Failed to fetch {symbol} {interval}: {e}")
                return pd.DataFrame()


def fetch_binance_oi(symbol: str, period: str = "15m", limit: int = 500) -> pd.Series:
    """Fetches historical open interest from Binance Futures."""
    cache_file = CACHE_DIR / f"{symbol}_oi_{period}_{limit}.csv"
    if cache_file.exists():
        try:
            df = pd.read_csv(cache_file)
            if "sumOpenInterest" in df.columns:
                return df["sumOpenInterest"].astype(float)
        except Exception:
            pass

    url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period={period}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            df = pd.DataFrame(data)
            if "sumOpenInterest" in df.columns:
                df["sumOpenInterest"] = df["sumOpenInterest"].astype(float)
                df.to_csv(cache_file, index=False)
                return df["sumOpenInterest"]
    except Exception as e:
        logger.warning(f"Could not fetch OI for {symbol}: {e}")
    return pd.Series(dtype=float)


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates all order flow, structural, and volatility features."""
    if len(df) < 15:
        return df

    df = df.copy()
    # 1. Delta & CVD
    df["delta"] = 2 * df["taker_buy_vol"] - df["volume"]
    df["cvd"] = df["delta"].cumsum()

    # 2. Delta Z-Score
    mean_d = df["delta"].rolling(20, min_periods=5).mean()
    std_d = df["delta"].rolling(20, min_periods=5).std().replace(0, 1e-6)
    df["delta_zscore"] = ((df["delta"] - mean_d) / std_d).fillna(0.0)

    # 3. RVOL & Candle Anatomy
    rolling_vol = df["volume"].rolling(20, min_periods=5).mean().replace(0, 1e-6)
    df["rvol"] = df["volume"] / rolling_vol

    c_range = (df["high"] - df["low"]).replace(0, 1e-6)
    df["candle_range"] = c_range
    df["candle_body"] = df["close"] - df["open"]
    df["close_location"] = (df["close"] - df["low"]) / c_range
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick_ratio"] = df["lower_wick"] / c_range
    df["upper_wick_ratio"] = df["upper_wick"] / c_range

    # 4. Indicators: RSI, EMA 20, ATR
    df["rsi"] = ConfluenceAnalyzer.compute_rsi(df["close"], 14)
    emas = ConfluenceAnalyzer.compute_emas(df["close"], [20, 50])
    df["ema_20"] = emas["ema_20"]
    df["ema_50"] = emas["ema_50"]
    atr, atr_pct = ConfluenceAnalyzer.compute_atr(df, 14)
    df["atr"] = atr
    df["atr_pct"] = atr_pct

    # 5. CVD Momentum
    df["cvd_slope"] = df["cvd"].diff(3).fillna(0.0)
    df["cvd_change_3"] = df["cvd"].diff(3).fillna(0.0)

    return df


def run_massive_backtest(
    min_score: float = 70.0,
    risk_reward: float = 2.0,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Runs the full battery of backtests across all selected symbols."""
    symbols = symbols or TOP_30_PAIRS
    tester = InstitutionalReplayBacktester(
        min_score=min_score,
        risk_reward_ratio=risk_reward,
        fee_rate=0.0005,
        slippage_rate=0.0002,
        max_bars_in_trade=48,  # 12 hours max hold on M15
    )

    results_by_symbol = {}
    all_trades = []

    logger.info(f"Starting Massive Backtest on {len(symbols)} pairs with Min Score {min_score} and R:R {risk_reward}...")

    for sym in symbols:
        df_m15 = fetch_binance_klines(sym, interval="15m", limit=1000)
        if len(df_m15) < 50:
            continue
        df_m15 = enrich_features(df_m15)

        df_m5 = fetch_binance_klines(sym, interval="5m", limit=1000)
        if len(df_m5) >= 15:
            df_m5 = enrich_features(df_m5)
        else:
            df_m5 = None

        oi_series = fetch_binance_oi(sym, period="15m", limit=500)

        res = tester.run(sym, df_m15=df_m15, df_m5=df_m5, oi_series=oi_series)
        results_by_symbol[sym] = res
        all_trades.extend(res.get("trades", []))

    # Aggregate global statistics
    total_trades = len(all_trades)
    if total_trades > 0:
        wins = [t for t in all_trades if t["is_win"]]
        losses = [t for t in all_trades if not t["is_win"]]
        global_wr = (len(wins) / total_trades) * 100.0

        gross_gains = sum(t["pnl_r"] for t in wins)
        gross_losses = abs(sum(t["pnl_r"] for t in losses))
        global_pf = (gross_gains / gross_losses) if gross_losses > 0 else 99.0
        global_expectancy = ((global_wr / 100.0) * risk_reward) - ((1.0 - (global_wr / 100.0)) * 1.0)
        net_pnl_r = sum(t["pnl_r"] for t in all_trades)
    else:
        global_wr = 0.0
        global_pf = 0.0
        global_expectancy = 0.0
        net_pnl_r = 0.0

    summary = {
        "min_score": min_score,
        "risk_reward_ratio": risk_reward,
        "symbols_tested": len(results_by_symbol),
        "total_trades": total_trades,
        "wins": len(wins) if total_trades > 0 else 0,
        "losses": len(losses) if total_trades > 0 else 0,
        "global_win_rate_pct": round(global_wr, 1),
        "global_profit_factor": round(global_pf, 2),
        "global_expectancy_r": round(global_expectancy, 2),
        "global_net_pnl_r": round(net_pnl_r, 2),
        "by_symbol": results_by_symbol,
    }

    return summary


if __name__ == "__main__":
    report = run_massive_backtest(min_score=70.0, risk_reward=2.0)
    print("\n" + "=" * 70)
    print("           RESULTADO DO BACKTEST MASSIVO INSTITUCIONAL")
    print("=" * 70)
    print(f"Pares Testados:       {report['symbols_tested']}")
    print(f"Total de Trades:      {report['total_trades']}")
    print(f"Taxa de Acerto (WR):  {report['global_win_rate_pct']}%")
    print(f"Profit Factor:        {report['global_profit_factor']}")
    print(f"Risco/Retorno Médio:  {report['risk_reward_ratio']}:1")
    print(f"Expectativa por Trade:{report['global_expectancy_r']} R")
    print(f"Resultado Líquido:    +{report['global_net_pnl_r']} R")
    print("=" * 70)
