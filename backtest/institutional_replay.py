"""Institutional Replay Backtesting Engine for Binance Futures USDⓈ-M.

Strict Execution Standards:
- Zero Look-Ahead Bias: Signals confirmed at candle [t] close execute ONLY at Open of candle [t+1].
- Realistic Trading Friction: Binance Futures standard taker fee (0.05%) + 0.02% slippage on entry & exit.
- Conservative Intrabar Evaluation: If a candle touches both Stop Loss and Take Profit,
  Stop Loss is ALWAYS triggered first.
- Evaluates Day Trade Reversal (CHoCH/BOS + Fibo Golden Zone + CVD Retest + VP Sweep + OI Trap + M5 Trigger).
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import logging

from indicators.market_structure import MarketStructure
from indicators.confluence import ConfluenceAnalyzer
from indicators.volume_profile import VolumeProfileAnalyzer
from indicators.oi_tracker import OpenInterestTracker
from signals.triple_engine import TripleConfluenceEngine

logger = logging.getLogger(__name__)


class InstitutionalReplayBacktester:
    """Rigorous walk-forward backtester for the Triple Confluence Institutional Strategy."""

    def __init__(
        self,
        min_score: float = 70.0,
        risk_reward_ratio: float = 2.0,
        fee_rate: float = 0.0005,      # 0.05% taker fee
        slippage_rate: float = 0.0002,  # 0.02% slippage
        max_bars_in_trade: int = 40,   # 10 hours max hold time on M15
    ):
        self.min_score = min_score
        self.risk_reward_ratio = risk_reward_ratio
        self.two_way_cost = 2 * (fee_rate + slippage_rate)
        self.max_bars_in_trade = max_bars_in_trade
        self.structure_engine = MarketStructure(pivot_left=3, pivot_right=2)

    def run(
        self,
        symbol: str,
        df_m15: pd.DataFrame,
        df_m5: Optional[pd.DataFrame] = None,
        oi_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """Runs the bar-by-bar walk-forward simulation."""
        n_bars = len(df_m15)
        if n_bars < 40:
            return {
                "symbol": symbol,
                "error": "Dados insuficientes (mínimo 40 barras M15 necessárias)",
                "total_trades": 0,
            }

        trades: List[Dict[str, Any]] = []
        active_trade: Optional[Dict[str, Any]] = None

        # Warm-up period of 25 bars for EMA, RSI, and pivots
        for i in range(25, n_bars - 1):
            curr_bar = df_m15.iloc[i]
            next_bar = df_m15.iloc[i + 1]

            # -----------------------------------------------------------------
            # 1. Active Trade Management
            # -----------------------------------------------------------------
            if active_trade is not None:
                side = active_trade["side"]
                entry_p = active_trade["entry_price"]
                stop_p = active_trade["stop_loss"]
                target_p = active_trade["take_profit"]
                bars_held = i - active_trade["entry_bar_idx"]

                trade_closed = False
                exit_price = None
                pnl_r = 0.0
                exit_reason = ""

                if side == "LONG":
                    # Conservative: Check Stop hit first on ambiguous bar
                    if next_bar["low"] <= stop_p:
                        exit_price = stop_p
                        pnl_r = -1.0
                        exit_reason = "STOP_LOSS"
                        trade_closed = True
                    elif next_bar["high"] >= target_p:
                        exit_price = target_p
                        pnl_r = self.risk_reward_ratio
                        exit_reason = "TAKE_PROFIT_2R"
                        trade_closed = True
                    elif bars_held >= self.max_bars_in_trade:
                        # Time stop: close at market open of timeout bar
                        exit_price = next_bar["open"]
                        risk = entry_p - stop_p
                        pnl_r = (exit_price - entry_p) / risk if risk > 0 else 0.0
                        exit_reason = "TIMEOUT_EXPIRATION"
                        trade_closed = True

                elif side == "SHORT":
                    if next_bar["high"] >= stop_p:
                        exit_price = stop_p
                        pnl_r = -1.0
                        exit_reason = "STOP_LOSS"
                        trade_closed = True
                    elif next_bar["low"] <= target_p:
                        exit_price = target_p
                        pnl_r = self.risk_reward_ratio
                        exit_reason = "TAKE_PROFIT_2R"
                        trade_closed = True
                    elif bars_held >= self.max_bars_in_trade:
                        exit_price = next_bar["open"]
                        risk = stop_p - entry_p
                        pnl_r = (entry_p - exit_price) / risk if risk > 0 else 0.0
                        exit_reason = "TIMEOUT_EXPIRATION"
                        trade_closed = True

                if trade_closed:
                    gross_ret = (exit_price - entry_p) / entry_p if side == "LONG" else (entry_p - exit_price) / entry_p
                    net_ret = gross_ret - self.two_way_cost
                    active_trade.update({
                        "exit_bar_idx": i + 1,
                        "exit_time": next_bar.get("open_time", i + 1),
                        "exit_price": round(exit_price, 4),
                        "pnl_r": round(pnl_r, 2),
                        "net_return_pct": round(net_ret * 100.0, 3),
                        "bars_held": bars_held,
                        "exit_reason": exit_reason,
                        "is_win": pnl_r > 0.0,
                    })
                    trades.append(active_trade)
                    active_trade = None
                continue

            # -----------------------------------------------------------------
            # 2. Evaluate Setup at Close of Bar [i]
            # -----------------------------------------------------------------
            slice_m15 = df_m15.iloc[: i + 1]
            curr_price = float(curr_bar["close"])

            # Market Structure (anti-repaint)
            struct_m15 = self.structure_engine.evaluate_structure(slice_m15)

            # Fibonacci Golden Zone anchored to active impulse wave
            fib_data = ConfluenceAnalyzer.compute_fibonacci_retracement(
                slice_m15,
                lookback=30,
                struct_info=struct_m15,
            )

            # Volume Profile
            vp_data = VolumeProfileAnalyzer.compute_volume_profile(
                slice_m15,
                lookback=min(100, len(slice_m15)),
                rows=24,
            )

            # Open Interest Traps
            oi_data = None
            if oi_series is not None and len(oi_series) > i:
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

            # M5 Tactical Slice strictly prior to or at bar [i] open_time
            m5_slice = None
            if df_m5 is not None and "open_time" in df_m5.columns and "open_time" in curr_bar:
                m5_slice = df_m5[df_m5["open_time"] <= curr_bar["open_time"]].tail(20)

            # Mock dummy H1 context dataframe if needed
            h1_dummy = slice_m15.iloc[::4] if len(slice_m15) >= 8 else slice_m15

            # Evaluate Day Trade Reversal Setup
            sig = TripleConfluenceEngine.evaluate_daytrade(
                m15_df=slice_m15,
                h1_df=h1_dummy,
                fib_data=fib_data,
                struct_m15=struct_m15,
                current_price=curr_price,
                spread_pct=0.0005,
                m5_df=m5_slice,
                vp_data=vp_data,
                oi_data=oi_data,
            )

            # Execution Gate: A+ Intraday Reversal Setup
            if sig.state == "DAYTRADE_ENTRY_A+" and sig.side in ["LONG", "SHORT"]:
                # Strict anti-lookahead execution at Open of Bar [i+1]
                entry_price = float(next_bar["open"])
                inval_price = float(sig.invalidation) if sig.invalidation else (curr_price * 0.995 if sig.side == "LONG" else curr_price * 1.005)

                if sig.side == "LONG":
                    risk = entry_price - inval_price
                    # Filter invalid risk geometries
                    if risk > 0 and (risk / entry_price) <= 0.035:
                        take_profit = entry_price + (risk * self.risk_reward_ratio)
                        active_trade = {
                            "symbol": symbol,
                            "side": "LONG",
                            "entry_bar_idx": i + 1,
                            "entry_time": next_bar.get("open_time", i + 1),
                            "entry_price": round(entry_price, 4),
                            "stop_loss": round(inval_price, 4),
                            "take_profit": round(take_profit, 4),
                            "risk": round(risk, 4),
                            "score": sig.score,
                            "reasons": sig.reasons[:3],
                        }
                elif sig.side == "SHORT":
                    risk = inval_price - entry_price
                    if risk > 0 and (risk / entry_price) <= 0.035:
                        take_profit = entry_price - (risk * self.risk_reward_ratio)
                        active_trade = {
                            "symbol": symbol,
                            "side": "SHORT",
                            "entry_bar_idx": i + 1,
                            "entry_time": next_bar.get("open_time", i + 1),
                            "entry_price": round(entry_price, 4),
                            "stop_loss": round(inval_price, 4),
                            "take_profit": round(take_profit, 4),
                            "risk": round(risk, 4),
                            "score": sig.score,
                            "reasons": sig.reasons[:3],
                        }

        # ---------------------------------------------------------------------
        # 3. Calculate Comprehensive Performance Metrics
        # ---------------------------------------------------------------------
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "symbol": symbol,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "avg_risk_reward": self.risk_reward_ratio,
                "expectancy_r": 0.0,
                "max_drawdown_r": 0.0,
                "trades": [],
            }

        wins = [t for t in trades if t["is_win"]]
        losses = [t for t in trades if not t["is_win"]]
        win_rate = (len(wins) / total_trades) * 100.0

        gross_gains = sum(t["pnl_r"] for t in wins)
        gross_losses = abs(sum(t["pnl_r"] for t in losses))
        profit_factor = (gross_gains / gross_losses) if gross_losses > 0 else (gross_gains if gross_gains > 0 else 1.0)

        # Mathematical Expectancy: E = (WR * AvgGain) - (LR * AvgLoss)
        loss_rate = 1.0 - (win_rate / 100.0)
        expectancy_r = ((win_rate / 100.0) * self.risk_reward_ratio) - (loss_rate * 1.0)

        # Max Drawdown calculation in R-units
        cum_r = 0.0
        peak_r = 0.0
        max_dd_r = 0.0
        for t in trades:
            cum_r += t["pnl_r"]
            if cum_r > peak_r:
                peak_r = cum_r
            dd = peak_r - cum_r
            if dd > max_dd_r:
                max_dd_r = dd

        return {
            "symbol": symbol,
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "avg_risk_reward": self.risk_reward_ratio,
            "expectancy_r": round(expectancy_r, 2),
            "max_drawdown_r": round(max_dd_r, 2),
            "net_pnl_r": round(cum_r, 2),
            "trades": trades,
        }
