"""Calibration script to isolate the True Institutional A+ Setup."""

import sys
from pathlib import Path
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


def run_experiment(
    require_retest_cvd: bool = True,
    require_m5_engulfing: bool = True,
    require_golden_zone: bool = True,
    min_score: float = 72.0,
    rr: float = 2.0,
    stop_atr_mult: float = 0.5,
    max_symbols: int = 15,
):
    struct_eng = MarketStructure(pivot_left=3, pivot_right=2)
    all_trades = []

    for sym in TOP_30_PAIRS[:max_symbols]:
        df_m15 = enrich_features(fetch_binance_klines(sym, "15m", 1000))
        if len(df_m15) < 50:
            continue
        df_m5 = enrich_features(fetch_binance_klines(sym, "5m", 1000))
        oi_series = fetch_binance_oi(sym, "15m", 500)

        active_trade = None
        for i in range(25, len(df_m15) - 1):
            curr_bar = df_m15.iloc[i]
            next_bar = df_m15.iloc[i + 1]

            if active_trade is not None:
                side = active_trade["side"]
                entry_p = active_trade["entry_p"]
                stop_p = active_trade["stop_p"]
                target_p = active_trade["target_p"]

                closed = False
                pnl_r = 0.0

                if side == "LONG":
                    if next_bar["low"] <= stop_p:
                        closed = True
                        pnl_r = -1.0
                    elif next_bar["high"] >= target_p:
                        closed = True
                        pnl_r = rr
                elif side == "SHORT":
                    if next_bar["high"] >= stop_p:
                        closed = True
                        pnl_r = -1.0
                    elif next_bar["low"] <= target_p:
                        closed = True
                        pnl_r = rr

                if closed:
                    active_trade["pnl_r"] = pnl_r
                    active_trade["is_win"] = pnl_r > 0
                    all_trades.append(active_trade)
                    active_trade = None
                continue

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
            if len(df_m5) > 0 and "open_time" in df_m5.columns:
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

            # Filtering conditions
            if sig.score < min_score or sig.side not in ["LONG", "SHORT"]:
                continue

            retest_cvd = sig.metrics.get("retest_cvd_div", False)
            in_gz = fib_data.get("in_golden_zone", False) or fib_data.get("distance_to_golden_zone_pct", 99.0) <= 0.5 or vp_data.get("val_sweep_bullish", False)
            m5_trig = sig.metrics.get("m5_engulfing", False)

            if require_retest_cvd and not retest_cvd:
                continue
            if require_golden_zone and not in_gz:
                continue
            if require_m5_engulfing and not m5_trig:
                continue

            # Invalidation with proper structural buffer
            entry_p = float(next_bar["open"])
            atr_val = float(curr_bar.get("atr", curr_price * 0.005))
            buffer = max(curr_price * 0.0015, atr_val * stop_atr_mult)

            if sig.side == "LONG":
                swing_lo = struct_m15.get("bos_origin_low") or struct_m15.get("choch_origin_low") or curr_bar["low"]
                stop_p = float(swing_lo) - buffer
                risk = entry_p - stop_p
                if 0 < risk <= entry_p * 0.035:
                    target_p = entry_p + (risk * rr)
                    active_trade = {"sym": sym, "side": "LONG", "entry_p": entry_p, "stop_p": stop_p, "target_p": target_p}
            elif sig.side == "SHORT":
                swing_hi = struct_m15.get("bos_origin_high") or struct_m15.get("choch_origin_high") or curr_bar["high"]
                stop_p = float(swing_hi) + buffer
                risk = stop_p - entry_p
                if 0 < risk <= entry_p * 0.035:
                    target_p = entry_p - (risk * rr)
                    active_trade = {"sym": sym, "side": "SHORT", "entry_p": entry_p, "stop_p": stop_p, "target_p": target_p}

    n = len(all_trades)
    if n == 0:
        print(f"Conf [CVD:{require_retest_cvd}, GZ:{require_golden_zone}, M5:{require_m5_engulfing}, Score:{min_score}]: 0 trades")
        return

    wins = sum(1 for t in all_trades if t["is_win"])
    wr = (wins / n) * 100.0
    net_r = sum(t["pnl_r"] for t in all_trades)
    pf = (wins * rr) / max(1, (n - wins))
    print(f"Conf [CVD={require_retest_cvd}, GZ={require_golden_zone}, M5={require_m5_engulfing}, Score>={min_score}, buf={stop_atr_mult}]: {n} trades | WR: {wr:.1f}% | Net: +{net_r:.1f}R | PF: {pf:.2f}")


if __name__ == "__main__":
    print("Testing Confluence combinations across 15 symbols...")
    run_experiment(require_retest_cvd=False, require_m5_engulfing=False, require_golden_zone=False, min_score=70.0)
    run_experiment(require_retest_cvd=True, require_m5_engulfing=False, require_golden_zone=False, min_score=70.0)
    run_experiment(require_retest_cvd=True, require_m5_engulfing=False, require_golden_zone=True, min_score=70.0)
    run_experiment(require_retest_cvd=True, require_m5_engulfing=True, require_golden_zone=True, min_score=70.0)
    run_experiment(require_retest_cvd=True, require_m5_engulfing=False, require_golden_zone=True, min_score=75.0, stop_atr_mult=0.8)
    run_experiment(require_retest_cvd=True, require_m5_engulfing=False, require_golden_zone=True, min_score=78.0, stop_atr_mult=0.8)

