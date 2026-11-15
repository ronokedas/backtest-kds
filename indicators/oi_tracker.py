"""Open Interest (OI) Trap & Liquidity Tracker for Binance Futures.

Detects institutional traps and retail positioning anomalies:
1. Short Trap (Bullish Reversal / Short Squeeze):
   - Price dips into support / Golden Zone / VAL.
   - Open Interest spikes up (retail FOMO shorting into the floor).
   - CVD exhibits bullish absorption divergence or positive delta (institutions absorbing via Bids).
   - Trapped shorts provide the buy-side fuel for explosive upward continuation!

2. Long Trap (Bearish Reversal / Long Squeeze):
   - Price pushes into resistance / VAH.
   - Open Interest spikes up (retail FOMO buying breakouts).
   - CVD exhibits bearish absorption divergence or negative delta (institutions selling into Asks).
   - Trapped longs provide sell-side cascading fuel!

3. Capitulation Flush:
   - Sharp price drop accompanied by massive OI collapse (forced liquidation cascade).
   - Once OI stabilizes and absorption appears, selling pressure is fully exhausted.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


class OpenInterestTracker:
    """Detects Open Interest anomalies, Short/Long Traps, and capitulation dynamics."""

    @staticmethod
    def detect_traps(
        current_oi: float,
        hist_oi: List[float],
        current_price: float,
        price_series: pd.Series,
        cvd_divergence: Dict[str, Any],
        delta_zscore: float = 0.0,
        in_support_zone: bool = False,
        in_resistance_zone: bool = False,
    ) -> Dict[str, Any]:
        """Calculates multi-window OI rates and detects institutional trap setups."""
        if not hist_oi or len(hist_oi) < 2 or current_oi <= 0:
            return {
                "oi_change_15m_pct": 0.0,
                "oi_change_60m_pct": 0.0,
                "short_trap_bullish": False,
                "long_trap_bearish": False,
                "capitulation_bullish": False,
                "oi_regime": "NEUTRAL",
                "trap_score_boost": 0.0,
                "reasons": [],
            }

        # 1. Calculate OI deltas across windows
        # Assuming hist_oi has 15-minute resolution samples
        oi_15m_prev = hist_oi[-1]
        oi_60m_prev = hist_oi[-4] if len(hist_oi) >= 4 else hist_oi[0]

        delta_15m_pct = ((current_oi - oi_15m_prev) / oi_15m_prev * 100.0) if oi_15m_prev > 0 else 0.0
        delta_60m_pct = ((current_oi - oi_60m_prev) / oi_60m_prev * 100.0) if oi_60m_prev > 0 else 0.0

        # Price changes
        p_prev_15m = float(price_series.iloc[-2]) if len(price_series) >= 2 else current_price
        p_delta_15m_pct = ((current_price - p_prev_15m) / p_prev_15m * 100.0) if p_prev_15m > 0 else 0.0

        short_trap = False
        long_trap = False
        capitulation = False
        reasons = []
        trap_score = 0.0

        # 2. Bullish Short Trap (Short Squeeze Ignition)
        # Conditions:
        # A) OI is surging (+1.2% in 15m or +2.5% in 60m)
        # B) Price is dropping or testing support / Golden Zone
        # C) Order flow shows absorption (CVD divergence, retest divergence, or delta absorbing)
        has_cvd_bull = (
            cvd_divergence.get("bullish_divergence", False)
            or cvd_divergence.get("bullish_absorption_divergence", False)
            or delta_zscore <= -1.2
        )

        if delta_15m_pct >= 1.0 or delta_60m_pct >= 2.0:
            if (p_delta_15m_pct <= 0.1 or in_support_zone) and has_cvd_bull:
                short_trap = True
                trap_score += 20.0
                reasons.append(
                    f"Short Trap Institucional: OI subiu {delta_15m_pct:+.2f}% no suporte com CVD divergente (venda do varejo absorvida por Bids — combustível para Short Squeeze)"
                )

        # 3. Bearish Long Trap (Long Squeeze Ignition)
        has_cvd_bear = (
            cvd_divergence.get("bearish_divergence", False)
            or cvd_divergence.get("bearish_absorption_divergence", False)
            or delta_zscore >= 1.2
        )

        if delta_15m_pct >= 1.0 or delta_60m_pct >= 2.0:
            if (p_delta_15m_pct >= -0.1 or in_resistance_zone) and has_cvd_bear:
                long_trap = True
                trap_score += 20.0
                reasons.append(
                    f"Long Trap Institucional: OI subiu {delta_15m_pct:+.2f}% na resistência com CVD exausto (compras de topo absorvidas por Asks — risco de Long Squeeze)"
                )

        # 4. Capitulation Flush (Exhaustion of liquidations)
        # Price dropped hard, OI collapsed (> 2.5% flush), but absorption emerged
        if (delta_15m_pct <= -2.0 or delta_60m_pct <= -4.0) and p_delta_15m_pct < -0.4:
            if has_cvd_bull or delta_zscore <= -1.5:
                capitulation = True
                trap_score += 15.0
                reasons.append(
                    f"Capitulação & Exaustão de Despejo: OI desabou {delta_15m_pct:+.2f}% com liquidações forçadas contidas por compradores passivos"
                )

        # Regime classification
        if short_trap:
            regime = "SHORT_TRAP_BULLISH"
        elif long_trap:
            regime = "LONG_TRAP_BEARISH"
        elif capitulation:
            regime = "CAPITULATION_REVERSAL"
        elif delta_15m_pct > 1.5:
            regime = "OI_SURGE"
        elif delta_15m_pct < -1.5:
            regime = "OI_UNWINDING"
        else:
            regime = "NEUTRAL"

        return {
            "oi_change_15m_pct": round(delta_15m_pct, 2),
            "oi_change_60m_pct": round(delta_60m_pct, 2),
            "short_trap_bullish": short_trap,
            "long_trap_bearish": long_trap,
            "capitulation_bullish": capitulation,
            "oi_regime": regime,
            "trap_score_boost": round(trap_score, 1),
            "reasons": reasons,
        }

