"""Test Institutional Trade Management (Breakeven at 1R, Target 2R, Full Confluence)."""

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


def run_institutional_trade_mgmt_test(
    symbols,
    min_score=75.0,
    target_r=2.0,
    be_trigger_r=1.0,  # Move stop to BE at 1R
    taker_fee=0.0005,
    slippage=0.0002,
):
    struct_eng = MarketStructure(pivot_left=3, pivot_right=2)
    two_way_friction = 2 * (taker_fee + slippage)
    all_trades = []

    for sym in symbols:
        df_m15 = enrich_features(fetch_binance_klines(sym, "15m", 1000))
        if len(df_m15) < 50:
            continue
        df_m5 = enrich_features(fetch_binance_klines(sym, "5m", 1000))
        oi_series = fetch_binance_oi(sym, "15m", 500)

        n_bars = len(df_m15)
        active_trade = None

        for i in range(30, n_bars - 1):
            curr_bar = df_m15.iloc[i]
            next_bar = df_m15.iloc[i + 1]

            if active_trade is not None:
                side = active_trade["side"]
                entry_p = active_trade["entry_p"]
                stop_p = active_trade["stop_p"]
                target_p = active_trade["target_p"]
                risk = active_trade["risk"]
                be_p = active_trade["be_p"]
                be_activated = active_trade["be_activated"]

                closed = False
                pnl_r = 0.0

                if side == "LONG":
                    # Check Breakeven activation (reached 1R)
                    if not be_activated and next_bar["high"] >= be_p:
                        active_trade["be_activated"] = True
                        active_trade["stop_p"] = entry_p * (1.0 + two_way_friction)  # Stop to BE + friction
                        stop_p = active_trade["stop_p"]

                    if next_bar["low"] <= stop_p:
                        closed = True
                        pnl_r = 0.0 if be_activated else -1.0
                    elif next_bar["high"] >= target_p:
                        closed = True
                        pnl_r = target_r

                elif side == "SHORT":
                    if not be_activated and next_bar["low"] <= be_p:
                        active_trade["be_activated"] = True
                        active_trade["stop_p"] = entry_p * (1.0 - two_way_friction)
                        stop_p = active_trade["stop_p"]

                    if next_bar["high"] >= stop_p:
                        closed = True
                        pnl_r = 0.0 if be_activated else -1.0
                    elif next_bar["low"] <= target_p:
                        closed = True
                        pnl_r = target_r

                if closed:
                    active_trade["pnl_r"] = pnl_r
                    active_trade["is_win"] = pnl_r > 0
                    active_trade["is_be"] = (pnl_r == 0.0)
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

            if sig.score < min_score or sig.side not in ["LONG", "SHORT"]:
                continue

            # Confluence gates
            has_cvd = sig.metrics.get("retest_cvd_div", False)
            in_gz = fib_data.get("in_golden_zone", False) or fib_data.get("distance_to_golden_zone_pct", 99.0) <= 0.6 or vp_data.get("val_sweep_bullish", False)
            has_m5 = sig.metrics.get("m5_engulfing", False)

            # High-precision A+ confluence:
            if not (has_cvd and in_gz):
                continue

            entry_p = float(next_bar["open"])

            if sig.side == "LONG":
                local_lo = float(slice_m15["low"].tail(3).min())
                stop_p = local_lo * 0.9985
                risk = entry_p - stop_p
                if 0 < risk <= entry_p * 0.025:
                    target_p = entry_p + (risk * target_r)
                    be_p = entry_p + (risk * be_trigger_r)
                    active_trade = {
                        "sym": sym, "side": "LONG", "entry_p": entry_p,
                        "stop_p": stop_p, "target_p": target_p,
                        "risk": risk, "be_p": be_p, "be_activated": False
                    }
            else:
                local_hi = float(slice_m15["high"].tail(3).max())
                stop_p = local_hi * 1.0015
                risk = stop_p - entry_p
                if 0 < risk <= entry_p * 0.025:
                    target_p = entry_p - (risk * target_r)
                    be_p = entry_p - (risk * be_trigger_r)
                    active_trade = {
                        "sym": sym, "side": "SHORT", "entry_p": entry_p,
                        "stop_p": stop_p, "target_p": target_p,
                        "risk": risk, "be_p": be_p, "be_activated": False
                    }

    n = len(all_trades)
    if n == 0:
        print("0 trades executed.")
        return

    wins = sum(1 for t in all_trades if t["is_win"])
    bes = sum(1 for t in all_trades if t["is_be"])
    losses = sum(1 for t in all_trades if t["pnl_r"] < 0)

    # Win rate of completed directional decisions (excluding BE) or total
    wr_total = (wins / n) * 100.0
    win_or_be = ((wins + bes) / n) * 100.0
    pf = (wins * target_r) / max(1, losses)
    net_r = sum(t["pnl_r"] for t in all_trades)

    print("\n" + "=" * 70)
    print("      RESULTADO INSTITUCIONAL: GESTÃO ATIVA COM BREAKEVEN NO 1R")
    print("=" * 70)
    print(f"Pares Testados:       {len(symbols)}")
    print(f"Total de Trades:      {n}")
    print(f"Trades com Gain (2R): {wins} ({wr_total:.1f}%)")
    print(f"Trades no Zero (BE):  {bes} ({bes/n*100:.1f}%)")
    print(f"Trades com Loss:      {losses} ({losses/n*100:.1f}%)")
    print(f"Assertividade Protegida (Gain + BE): {win_or_be:.1f}%")
    print(f"Profit Factor:        {pf:.2f}")
    print(f"Resultado Líquido:    +{net_r:.1f} R")
    print("=" * 70)


if __name__ == "__main__":
    run_institutional_trade_mgmt_test(TOP_30_PAIRS[:15], min_score=75.0, target_r=2.0, be_trigger_r=1.0)

