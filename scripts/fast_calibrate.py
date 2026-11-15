"""High-Performance Institutional Confluence Calibrator for 30 Binance Pairs.

Precomputes multi-timeframe indicators and forward price trajectories once,
then benchmarks confluence combinations to meet:
- Win Rate >= 70%
- Risk/Reward >= 2.0 (2:1)
- Sample size >= 20 trades per symbol (>= 600 trades total across 30 pairs)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from scripts.run_massive_backtest import fetch_binance_klines, enrich_features, fetch_binance_oi, TOP_30_PAIRS
from indicators.market_structure import MarketStructure
from indicators.confluence import ConfluenceAnalyzer
from indicators.volume_profile import VolumeProfileAnalyzer
from indicators.oi_tracker import OpenInterestTracker
from signals.triple_engine import TripleConfluenceEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Calibrator")


def precompute_market_signals(symbols: List[str]) -> List[Dict[str, Any]]:
    """Precomputes all bars and indicators across symbols for ultra-fast replay."""
    struct_eng = MarketStructure(pivot_left=3, pivot_right=2)
    dataset = []

    logger.info(f"Precomputing indicators across {len(symbols)} symbols...")
    for sym_idx, sym in enumerate(symbols, 1):
        df_m15 = fetch_binance_klines(sym, "15m", 1000)
        if len(df_m15) < 60:
            continue
        df_m15 = enrich_features(df_m15)
        df_m5 = fetch_binance_klines(sym, "5m", 1000)
        if len(df_m5) >= 20:
            df_m5 = enrich_features(df_m5)
        else:
            df_m5 = None
        oi_series = fetch_binance_oi(sym, "15m", 500)

        n_bars = len(df_m15)
        for i in range(30, n_bars - 25):
            curr_bar = df_m15.iloc[i]
            next_bar = df_m15.iloc[i + 1]
            future_bars = df_m15.iloc[i + 1 : min(n_bars, i + 41)]

            slice_m15 = df_m15.iloc[: i + 1]
            curr_price = float(curr_bar["close"])

            struct_m15 = struct_eng.evaluate_structure(slice_m15)
            fib_data = ConfluenceAnalyzer.compute_fibonacci_retracement(slice_m15, lookback=30, struct_info=struct_m15)
            vp_data = VolumeProfileAnalyzer.compute_volume_profile(slice_m15, lookback=min(100, len(slice_m15)), rows=24)

            oi_data = None
            if len(oi_series) > i:
                curr_oi = float(oi_series.iloc[i])
                hist_oi = [float(x) for x in oi_series.iloc[max(0, i - 12) : i]]
                oi_data = OpenInterestTracker.detect_traps(
                    current_oi=curr_oi,
                    hist_oi=hist_oi,
                    current_price=curr_price,
                    price_series=slice_m15["close"],
                    cvd_divergence=ConfluenceAnalyzer.detect_retest_cvd_divergence(slice_m15, struct_m15),
                    delta_zscore=float(curr_bar.get("delta_zscore", 0.0)),
                    in_support_zone=fib_data.get("in_golden_zone", False) or vp_data.get("val_sweep_bullish", False),
                    in_resistance_zone=vp_data.get("vah_sweep_bearish", False),
                )

            m5_slice = None
            if df_m5 is not None and "open_time" in df_m5.columns:
                m5_slice = df_m5[df_m5["open_time"] <= curr_bar["open_time"]].tail(20)

            sig = TripleConfluenceEngine.evaluate_daytrade(
                m15_df=slice_m15,
                h1_df=slice_m15.iloc[::4],
                fib_data=fib_data,
                struct_m15=struct_m15,
                current_price=curr_price,
                spread_pct=0.0005,
                m5_df=m5_slice,
                vp_data=vp_data,
                oi_data=oi_data,
            )

            # Record candidate setup
            dataset.append({
                "symbol": sym,
                "bar_idx": i,
                "entry_open": float(next_bar["open"]),
                "curr_close": curr_price,
                "curr_low": float(curr_bar["low"]),
                "curr_high": float(curr_bar["high"]),
                "atr": float(curr_bar.get("atr", curr_price * 0.005)),
                "side": sig.side,
                "score": sig.score,
                "state": sig.state,
                "invalidation": float(sig.invalidation) if sig.invalidation else curr_price * 0.995,
                "target_1": float(sig.target_1) if sig.target_1 else curr_price * 1.02,
                "retest_cvd": sig.metrics.get("retest_cvd_div", False),
                "in_golden_zone": fib_data.get("in_golden_zone", False) or fib_data.get("distance_to_golden_zone_pct", 99.0) <= 0.6,
                "val_sweep": vp_data.get("val_sweep_bullish", False),
                "vah_sweep": vp_data.get("vah_sweep_bearish", False),
                "short_trap": oi_data.get("short_trap_bullish", False) if oi_data else False,
                "long_trap": oi_data.get("long_trap_bearish", False) if oi_data else False,
                "m5_engulfing": sig.metrics.get("m5_engulfing", False),
                "future_highs": future_bars["high"].values,
                "future_lows": future_bars["low"].values,
                "future_opens": future_bars["open"].values,
            })
        if sym_idx % 5 == 0 or sym_idx == len(symbols):
            logger.info(f"Processed {sym_idx}/{len(symbols)} symbols. Dataset size: {len(dataset)} bars.")

    return dataset


def evaluate_policy(
    dataset: List[Dict[str, Any]],
    min_score: float,
    require_retest_cvd: bool = False,
    require_golden_zone_or_sweep: bool = False,
    require_m5: bool = False,
    require_trap: bool = False,
    stop_buffer_atr: float = 0.4,
    rr: float = 2.0,
    max_risk_pct: float = 0.035,
) -> Dict[str, Any]:
    """Evaluates a parameter combination across the precomputed dataset with anti-lookahead rules."""
    trades = []
    active_by_sym = {}

    for item in dataset:
        sym = item["symbol"]
        bar_idx = item["bar_idx"]

        # Check if symbol already has active trade in execution
        if sym in active_by_sym:
            if bar_idx <= active_by_sym[sym]:
                continue
            else:
                del active_by_sym[sym]

        if item["side"] not in ["LONG", "SHORT"]:
            continue
        if item["score"] < min_score:
            continue

        if require_retest_cvd and not item["retest_cvd"]:
            continue
        if require_golden_zone_or_sweep:
            gz_or_swp = item["in_golden_zone"] or (item["val_sweep"] if item["side"] == "LONG" else item["vah_sweep"])
            if not gz_or_swp:
                continue
        if require_m5 and not item["m5_engulfing"]:
            continue
        if require_trap:
            trap_active = item["short_trap"] if item["side"] == "LONG" else item["long_trap"]
            if not trap_active:
                continue

        # Invalidation & Target geometry
        entry_p = item["entry_open"]
        atr = item["atr"]
        buf = max(entry_p * 0.0015, atr * stop_buffer_atr)

        if item["side"] == "LONG":
            stop_p = item["invalidation"] - buf
            risk = entry_p - stop_p
            if risk <= 0 or (risk / entry_p) > max_risk_pct:
                continue
            target_p = entry_p + (risk * rr)
        else:
            stop_p = item["invalidation"] + buf
            risk = stop_p - entry_p
            if risk <= 0 or (risk / entry_p) > max_risk_pct:
                continue
            target_p = entry_p - (risk * rr)

        # Walk-forward through future bars
        future_h = item["future_highs"]
        future_l = item["future_lows"]
        pnl_r = 0.0
        exit_bar = len(future_h)

        for b in range(len(future_h)):
            hi = future_h[b]
            lo = future_l[b]

            if item["side"] == "LONG":
                # Conservative: Stop loss checked first on intrabar collision
                if lo <= stop_p:
                    pnl_r = -1.0
                    exit_bar = b
                    break
                elif hi >= target_p:
                    pnl_r = rr
                    exit_bar = b
                    break
            else:
                if hi >= stop_p:
                    pnl_r = -1.0
                    exit_bar = b
                    break
                elif lo <= target_p:
                    pnl_r = rr
                    exit_bar = b
                    break

        active_by_sym[sym] = bar_idx + exit_bar
        trades.append({
            "symbol": sym,
            "side": item["side"],
            "score": item["score"],
            "pnl_r": pnl_r,
            "is_win": pnl_r > 0,
        })

    n = len(trades)
    if n == 0:
        return {"n": 0, "wr": 0.0, "pf": 0.0, "net_r": 0.0}

    wins = sum(1 for t in trades if t["is_win"])
    wr = (wins / n) * 100.0
    net_r = sum(t["pnl_r"] for t in trades)
    gross_win = wins * rr
    gross_loss = max(1, n - wins)
    pf = gross_win / gross_loss

    # Check distribution per symbol
    sym_counts = pd.Series([t["symbol"] for t in trades]).value_counts()
    min_per_sym = sym_counts.min() if len(sym_counts) > 0 else 0
    avg_per_sym = n / len(dataset) if len(dataset) > 0 else 0

    return {
        "n": n,
        "wr": round(wr, 1),
        "pf": round(pf, 2),
        "net_r": round(net_r, 1),
        "trades": trades,
        "symbols_count": len(sym_counts),
        "min_per_sym": int(min_per_sym),
    }


if __name__ == "__main__":
    dataset = precompute_market_signals(TOP_30_PAIRS)
    
    print("\n" + "=" * 80)
    print("        GRADE DE PESQUISA QUANTITATIVA — CONFLUÊNCIAS INSTITUCIONAIS")
    print("=" * 80)
    print(f"{'Configuração':<45} | {'Trades':<7} | {'Win Rate':<9} | {'Profit Factor':<13} | {'Net (R)':<8}")
    print("-" * 80)

    configs = [
        {"name": "Base (Score >= 70, sem filtros)", "score": 70.0, "cvd": False, "gz": False, "m5": False, "trap": False, "buf": 0.2},
        {"name": "CVD Retest Div (Score >= 70)", "score": 70.0, "cvd": True, "gz": False, "m5": False, "trap": False, "buf": 0.3},
        {"name": "CVD Retest + Golden Zone / Sweep", "score": 70.0, "cvd": True, "gz": True, "m5": False, "trap": False, "buf": 0.3},
        {"name": "CVD Retest + GZ + M5 Engulfing", "score": 70.0, "cvd": True, "gz": True, "m5": True, "trap": False, "buf": 0.4},
        {"name": "CVD + GZ + M5 + Score >= 75", "score": 75.0, "cvd": True, "gz": True, "m5": True, "trap": False, "buf": 0.5},
        {"name": "A+ Institutional (Score >= 78 + Buffer ATR 0.6)", "score": 78.0, "cvd": True, "gz": True, "m5": True, "trap": False, "buf": 0.6},
        {"name": "A+ Sniper + OI Trap / Absorption", "score": 75.0, "cvd": True, "gz": True, "m5": False, "trap": True, "buf": 0.5},
        {"name": "High Precision Retest (Score >= 80)", "score": 80.0, "cvd": True, "gz": True, "m5": False, "trap": False, "buf": 0.5},
    ]

    for c in configs:
        res = evaluate_policy(
            dataset=dataset,
            min_score=c["score"],
            require_retest_cvd=c["cvd"],
            require_golden_zone_or_sweep=c["gz"],
            require_m5=c["m5"],
            require_trap=c["trap"],
            stop_buffer_atr=c["buf"],
            rr=2.0,
        )
        print(f"{c['name']:<45} | {res['n']:<7} | {res['wr']:<8}% | {res['pf']:<13} | +{res['net_r']}R")
    print("=" * 80)

