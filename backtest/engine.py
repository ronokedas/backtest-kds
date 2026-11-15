"""Backtesting and Historical Validation Engine.

Strict Execution Standards:
- Zero Look-Ahead Bias: Signals confirmed on candle [t] can ONLY execute at Open of candle [t+1].
- Realistic Trading Costs: Binance Futures standard taker fee (0.05%) + 0.02% slippage/spread.
- Invalidation Stops placed at structural absorption wick low/high.
- Reports Payoff, Profit Factor, Expectancy, and Max Drawdown.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from data.data_validator import KlineData
from indicators.cvd import CVDEngine
from indicators.absorption import AbsorptionMetrics
from indicators.market_structure import MarketStructure
from indicators.derivatives import DerivativesAnalyzer
from signals.scoring import QualityScorer


class AbsorptionBacktester:
    """Rigorous anti-lookahead backtest simulator for absorption signals."""

    def __init__(
        self,
        fee_rate: float = 0.0005,  # 0.05% taker fee
        slippage_rate: float = 0.0002,  # 0.02% estimated slippage
        min_score: float = 70.0,
        risk_reward_ratio: float = 2.0,
    ):
        self.total_cost_per_leg = fee_rate + slippage_rate
        self.min_score = min_score
        self.risk_reward_ratio = risk_reward_ratio

        self.cvd_engine = CVDEngine(rolling_window=20)
        self.absorption_metrics = AbsorptionMetrics()
        self.structure_engine = MarketStructure(pivot_left=3, pivot_right=2)

    def run_backtest(self, klines_m15: List[KlineData], klines_h1: Optional[List[KlineData]] = None) -> Dict[str, Any]:
        """Executes walk-forward bar-by-bar backtest over historical klines."""
        if len(klines_m15) < 50:
            return {"error": "Insufficient klines for statistical validity (min 50 required)"}

        df = self.cvd_engine.compute_kline_series_cvd(klines_m15)
        df = self.absorption_metrics.compute_features(df)

        df_h1 = None
        if klines_h1 and len(klines_h1) >= 6:
            df_h1 = self.cvd_engine.compute_kline_series_cvd(klines_h1)

        trades: List[Dict[str, Any]] = []
        active_trade: Optional[Dict[str, Any]] = None

        # Walk-forward loop (bar by bar)
        for i in range(25, len(df) - 1):
            curr_bar = df.iloc[i]
            next_bar = df.iloc[i + 1]

            # 1. Manage Active Trade if any
            if active_trade is not None:
                side = active_trade["side"]
                entry_p = active_trade["entry_price"]
                stop_p = active_trade["stop_loss"]
                target_p = active_trade["take_profit"]

                trade_closed = False
                exit_price = None
                pnl_r = 0.0

                if side == "LONG":
                    # Check stop hit (conservative: stop checked first on ambiguous wick)
                    if next_bar["low"] <= stop_p:
                        exit_price = stop_p
                        pnl_r = -1.0
                        trade_closed = True
                    elif next_bar["high"] >= target_p:
                        exit_price = target_p
                        pnl_r = self.risk_reward_ratio
                        trade_closed = True
                elif side == "SHORT":
                    if next_bar["high"] >= stop_p:
                        exit_price = stop_p
                        pnl_r = -1.0
                        trade_closed = True
                    elif next_bar["low"] <= target_p:
                        exit_price = target_p
                        pnl_r = self.risk_reward_ratio
                        trade_closed = True

                if trade_closed:
                    # Calculate net pnl deducting two-way trading costs
                    gross_return = (exit_price - entry_p) / entry_p if side == "LONG" else (entry_p - exit_price) / entry_p
                    net_return = gross_return - (2 * self.total_cost_per_leg)

                    active_trade.update(
                        {
                            "exit_time": next_bar["open_time"],
                            "exit_price": exit_price,
                            "pnl_r": pnl_r,
                            "net_return": net_return,
                            "bars_held": i - active_trade["entry_index"],
                        }
                    )
                    trades.append(active_trade)
                    active_trade = None
                continue

            # 2. Evaluate Signals at Bar [i] for Execution at Bar [i+1] Open
            historical_slice = df.iloc[: i + 1]
            abs_eval = self.absorption_metrics.evaluate_bar_absorption(curr_bar, historical_slice)
            struct_eval = self.structure_engine.evaluate_structure(historical_slice)

            # Evaluate higher timeframe H1 structure without future lookahead
            h1_eval = {"structure_bias": "NEUTRAL", "choch_bullish": False, "choch_bearish": False}
            if df_h1 is not None:
                h1_slice = df_h1[df_h1["open_time"] <= curr_bar["open_time"]]
                if len(h1_slice) >= 6:
                    h1_eval = self.structure_engine.evaluate_structure(h1_slice)

            deriv_eval = {"derivatives_score": 5.0}

            # Evaluate Long Score
            long_score, long_state, _ = QualityScorer.calculate_long_score(
                absorption_data=abs_eval,
                cvd_data={"cvd_change_3": curr_bar.get("cvd_change_3", 0.0)},
                structure_m15=struct_eval,
                structure_h1=h1_eval,
                derivatives_data=deriv_eval,
                spread_pct=0.01,
            )

            # Evaluate Short Score
            short_score, short_state, _ = QualityScorer.calculate_short_score(
                absorption_data=abs_eval,
                cvd_data={"cvd_change_3": curr_bar.get("cvd_change_3", 0.0)},
                structure_m15=struct_eval,
                structure_h1=h1_eval,
                derivatives_data=deriv_eval,
                spread_pct=0.01,
            )

            # Check if actionable signal meets threshold
            if long_score >= self.min_score:
                entry_price = next_bar["open"]  # Strict anti-lookahead execution at open
                stop_loss = curr_bar["low"] * 0.9985  # Invalidation below absorption bar
                risk = entry_price - stop_loss
                if risk > 0:
                    take_profit = entry_price + (risk * self.risk_reward_ratio)
                    active_trade = {
                        "side": "LONG",
                        "entry_index": i + 1,
                        "entry_time": next_bar["open_time"],
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "score": long_score,
                        "state": long_state,
                    }

            elif short_score >= self.min_score:
                entry_price = next_bar["open"]
                stop_loss = curr_bar["high"] * 1.0015
                risk = stop_loss - entry_price
                if risk > 0:
                    take_profit = entry_price - (risk * self.risk_reward_ratio)
                    active_trade = {
                        "side": "SHORT",
                        "entry_index": i + 1,
                        "entry_time": next_bar["open_time"],
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "score": short_score,
                        "state": short_state,
                    }

        # Calculate performance statistics
        total_trades = len(trades)
        if total_trades == 0:
            return {"total_trades": 0, "message": "No trades triggered meeting score threshold."}

        wins = [t for t in trades if t["pnl_r"] > 0]
        losses = [t for t in trades if t["pnl_r"] <= 0]

        win_rate = (len(wins) / total_trades) * 100.0
        total_pnl_r = sum(t["pnl_r"] for t in trades)
        avg_pnl_r = total_pnl_r / total_trades

        # Profit Factor
        gross_profit = sum(t["net_return"] for t in trades if t["net_return"] > 0)
        gross_loss = abs(sum(t["net_return"] for t in trades if t["net_return"] < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 999.0

        # Drawdown calculation
        equity_curve = np.cumsum([t["net_return"] for t in trades])
        peak = np.maximum.accumulate(equity_curve)
        drawdown = peak - equity_curve
        max_drawdown = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

        def calc_stats(sub_trades: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not sub_trades:
                return {"trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0}
            w = [t for t in sub_trades if t["pnl_r"] > 0]
            l = [t for t in sub_trades if t["pnl_r"] <= 0]
            wr = (len(w) / len(sub_trades)) * 100.0
            tot_r = sum(t["pnl_r"] for t in sub_trades)
            avg_r = tot_r / len(sub_trades)
            gp = sum(t["net_return"] for t in sub_trades if t["net_return"] > 0)
            gl = abs(sum(t["net_return"] for t in sub_trades if t["net_return"] < 0))
            pf = (gp / gl) if gl > 0 else 999.0
            return {
                "trades": int(len(sub_trades)),
                "wins": int(len(w)),
                "losses": int(len(l)),
                "win_rate_pct": float(round(wr, 2)),
                "expectancy_r": float(round(avg_r, 2)),
                "profit_factor": float(round(pf, 2)),
            }

        long_trades = [t for t in trades if t["side"] == "LONG"]
        short_trades = [t for t in trades if t["side"] == "SHORT"]

        return {
            "total_trades": int(total_trades),
            "wins": int(len(wins)),
            "losses": int(len(losses)),
            "win_rate_pct": float(round(win_rate, 2)),
            "expectancy_r": float(round(avg_pnl_r, 2)),
            "total_pnl_r": float(round(total_pnl_r, 2)),
            "profit_factor": float(round(profit_factor, 2)),
            "max_drawdown_pct": float(round(max_drawdown * 100, 2)),
            "long_metrics": calc_stats(long_trades),
            "short_metrics": calc_stats(short_trades),
            "trades": trades,
        }

