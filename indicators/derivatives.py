"""Derivatives Context Engine (USDⓈ-M Futures).

Processes:
- Open Interest (OI) changes.
- Taker Long/Short ratio.
- Funding rate.
- Contextual combination of Price, CVD, and OI.
"""

from typing import Dict, Any, List, Optional


class DerivativesAnalyzer:
    """Interprets derivatives market data within the absorption context."""

    @staticmethod
    def analyze_derivatives_context(
        current_oi: float,
        hist_oi_records: List[Dict[str, Any]],
        funding_rate: Optional[float],
        price_change_pct: float,
        delta_sign: float,
    ) -> Dict[str, Any]:
        """Combines Open Interest variation, funding rate, and price movement.

        Returns interpretation and context score contribution.
        """
        oi_change_pct = 0.0
        if hist_oi_records:
            prev_record = hist_oi_records[-2] if len(hist_oi_records) >= 2 else hist_oi_records[-1]
            if isinstance(prev_record, dict):
                prev_oi = float(prev_record.get("sumOpenInterest", prev_record.get("openInterest", current_oi)))
            else:
                prev_oi = float(prev_record)
            if prev_oi > 0:
                oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100.0

        # Context diagnosis:
        context_desc = "NEUTRAL"
        derivatives_score = 5.0  # Base neutral

        if price_change_pct < -0.1:
            if oi_change_pct < -0.5:
                # Price dropping + OI dropping: Long liquidation / deleveraging
                context_desc = "LONG_LIQUIDATION_UNWINDING"
                # If delta is negative but price stalls (absorption), liquidation cascade absorbed by bids
                derivatives_score = 8.5
            elif oi_change_pct > 0.5:
                # Price dropping + OI rising: Aggressive short buildup
                context_desc = "SHORT_BUILDUP"
                derivatives_score = 6.0
        elif price_change_pct > 0.1:
            if oi_change_pct > 0.5:
                # Price rising + OI rising: Long buildup
                context_desc = "LONG_BUILDUP"
                derivatives_score = 6.0
            elif oi_change_pct < -0.5:
                # Price rising + OI dropping: Short covering
                context_desc = "SHORT_COVERING"
                derivatives_score = 7.5

        return {
            "open_interest": current_oi,
            "oi_change_pct": oi_change_pct,
            "funding_rate": funding_rate,
            "derivatives_context": context_desc,
            "derivatives_score": derivatives_score,
        }

