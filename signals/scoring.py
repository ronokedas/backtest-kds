"""Quality Scoring Engine for Long and Short Absorption Signals.

Calculates a 0-100 score according to the 8 weighted microstructure components
specified in Section 7 of the master architecture.
"""

from typing import Dict, Any, Tuple, Optional


class QualityScorer:
    """Calculates weighted 0-100 absorption score."""

    WEIGHTS = {
        "flow_inefficiency": 25.0,
        "cvd_delta": 15.0,
        "price_response": 15.0,
        "m15_structure": 15.0,
        "h1_h4_confirmation": 10.0,
        "oi_liquidations": 10.0,
        "volume_liquidity": 5.0,
        "structural_zone": 5.0,
    }

    @classmethod
    def calculate_long_score(
        cls,
        absorption_data: Dict[str, Any],
        cvd_data: Dict[str, Any],
        structure_m15: Dict[str, Any],
        structure_h1: Dict[str, Any],
        derivatives_data: Dict[str, Any],
        spread_pct: float,
        structure_h4: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str, Dict[str, float]]:
        """Calculates Long Absorption Score (0-100)."""
        components: Dict[str, float] = {}

        # 1. Flow Inefficiency (0-25)
        # Heavy aggressive selling that fails to drop the price, OR Volume Dry-Up on structural retest
        buy_abs_score = absorption_data.get("buyer_absorption_score", 0.0)
        flow_ratio = buy_abs_score / 100.0
        is_retest = structure_m15.get("is_choch_retest", False) or structure_m15.get("in_golden_zone", False)
        rvol = absorption_data.get("rvol", 1.0)

        if is_retest and structure_m15.get("choch_bullish", False):
            if rvol <= 0.9:
                # Volume Dry-Up / Seller Exhaustion on Fibonacci / CHoCH retest
                flow_ratio = max(flow_ratio, 0.80)
            elif absorption_data.get("lower_wick_ratio", 0.0) >= 0.25:
                flow_ratio = max(flow_ratio, 0.70)

        components["flow_inefficiency"] = flow_ratio * cls.WEIGHTS["flow_inefficiency"]

        # 2. CVD and Delta (0-15)
        # CVD stabilizing or curving up after selling climax
        delta_zscore = absorption_data.get("delta_zscore", 0.0)
        cvd_change = cvd_data.get("cvd_change_3", 0.0)
        cvd_score = 0.0
        if delta_zscore <= -1.5:  # Significant selling climax
            cvd_score += 0.5
        if cvd_change >= 0:  # CVD halted falling or ticking up
            cvd_score += 0.5
        if is_retest and structure_m15.get("choch_bullish", False) and abs(delta_zscore) < 1.2:
            cvd_score = max(cvd_score, 0.8)  # Delta stabilization on pullback
        components["cvd_delta"] = min(1.0, cvd_score) * cls.WEIGHTS["cvd_delta"]

        # 3. Price Response & Retest (0-15)
        # Long lower wick or green close despite selling
        price_resp = 0.0
        lower_wick = absorption_data.get("lower_wick_ratio", 0.0)
        if lower_wick >= 0.35:
            price_resp += 0.6
        elif lower_wick >= 0.22:
            price_resp += 0.4
        if absorption_data.get("buyer_absorption_type") == "EXHAUSTION_REVERSAL":
            price_resp += 0.4
        if is_retest and structure_m15.get("choch_bullish", False) and lower_wick >= 0.20:
            price_resp = max(price_resp, 0.8)
        components["price_response"] = min(1.0, price_resp) * cls.WEIGHTS["price_response"]

        # 4. M15 Structure CHoCH / BOS / Liquidity Sweep (0-15)
        m15_score = 0.0
        if structure_m15.get("choch_bullish", False):
            m15_score = 1.0
        elif structure_m15.get("liquidity_sweep_bullish", False):
            m15_score = 0.8  # Strong institutional rejection of lower liquidity
        elif structure_m15.get("structure_bias") == "BULLISH":
            m15_score = 0.5
        components["m15_structure"] = m15_score * cls.WEIGHTS["m15_structure"]

        # 5. H1 / H4 Confirmation (0-10)
        h1_score = 0.0
        h1_bias = structure_h1.get("structure_bias", "NEUTRAL")
        if h1_bias == "BULLISH":
            h1_score = 0.85
        elif h1_bias == "NEUTRAL":
            h1_score = 0.5
        elif h1_bias == "BEARISH":
            h1_score = 0.0  # Counter-trend penalty: no points if higher timeframe opposes
            
        if structure_h1.get("choch_bullish", False):
            h1_score = 1.0

        if structure_h4:
            h4_bias = structure_h4.get("structure_bias", "NEUTRAL")
            if h4_bias == "BULLISH":
                h1_score = min(1.0, h1_score + 0.15)
            elif h4_bias == "BEARISH":
                h1_score = max(0.0, h1_score - 0.25)

        components["h1_h4_confirmation"] = h1_score * cls.WEIGHTS["h1_h4_confirmation"]

        # 6. OI and Liquidations (0-10)
        deriv_score = derivatives_data.get("derivatives_score", 5.0) / 10.0
        if derivatives_data.get("long_liquidations_absorbed", False):
            deriv_score = min(1.0, deriv_score + 0.2)
        components["oi_liquidations"] = deriv_score * cls.WEIGHTS["oi_liquidations"]

        # 7. Volume, Spread, Liquidity, and Order Book Imbalance (0-5)
        rvol = absorption_data.get("rvol", 1.0)
        obi = absorption_data.get("orderbook_imbalance", 0.0)
        liq_score = 0.4
        if rvol >= 1.2 and spread_pct <= 0.03:
            liq_score = 0.8
            if obi > 0.15:  # Order Book Imbalance in favor of bids (passive absorption)
                liq_score = 1.0
        elif rvol >= 1.0 and spread_pct <= 0.04:
            liq_score = 0.6
        components["volume_liquidity"] = liq_score * cls.WEIGHTS["volume_liquidity"]

        # 8. Structural Zone (0-5) - Real proximity calculation
        if "support_proximity" in structure_m15:
            zone_score = float(structure_m15["support_proximity"])
        elif structure_m15.get("choch_bullish") or structure_m15.get("liquidity_sweep_bullish"):
            zone_score = 0.8
        else:
            zone_score = 0.3

        if absorption_data.get("vwap_confluence_bullish", False):
            zone_score = min(1.0, zone_score + 0.2)

        components["structural_zone"] = zone_score * cls.WEIGHTS["structural_zone"]

        total_score = sum(components.values())

        # Determine state
        state = "NO_SIGNAL"
        if total_score >= 85.0 and structure_m15.get("choch_bullish", False):
            state = "CONFIRMED_REVERSAL"
        elif total_score >= 70.0:
            state = "POTENTIAL_ABSORPTION"
        elif total_score >= 55.0:
            state = "PRE_ALERT"
        elif total_score >= 40.0:
            state = "OBSERVATION"

        return round(total_score, 1), state, components

    @classmethod
    def calculate_short_score(
        cls,
        absorption_data: Dict[str, Any],
        cvd_data: Dict[str, Any],
        structure_m15: Dict[str, Any],
        structure_h1: Dict[str, Any],
        derivatives_data: Dict[str, Any],
        spread_pct: float,
        structure_h4: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, str, Dict[str, float]]:
        """Calculates Short Absorption Score (0-100)."""
        components: Dict[str, float] = {}

        # 1. Flow Inefficiency (0-25)
        # Heavy aggressive buying that fails to raise the price, OR Volume Dry-Up on structural retest
        sell_abs_score = absorption_data.get("seller_absorption_score", 0.0)
        flow_ratio = sell_abs_score / 100.0
        is_retest = structure_m15.get("is_choch_retest", False) or structure_m15.get("in_golden_zone", False)
        rvol = absorption_data.get("rvol", 1.0)

        if is_retest and structure_m15.get("choch_bearish", False):
            if rvol <= 0.9:
                # Volume Dry-Up / Buyer Exhaustion on Fibonacci / CHoCH retest
                flow_ratio = max(flow_ratio, 0.80)
            elif absorption_data.get("upper_wick_ratio", 0.0) >= 0.25:
                flow_ratio = max(flow_ratio, 0.70)

        components["flow_inefficiency"] = flow_ratio * cls.WEIGHTS["flow_inefficiency"]

        # 2. CVD and Delta (0-15)
        delta_zscore = absorption_data.get("delta_zscore", 0.0)
        cvd_change = cvd_data.get("cvd_change_3", 0.0)
        cvd_score = 0.0
        if delta_zscore >= 1.5:  # Significant buying climax
            cvd_score += 0.5
        if cvd_change <= 0:  # CVD halted rising or ticking down
            cvd_score += 0.5
        if is_retest and structure_m15.get("choch_bearish", False) and abs(delta_zscore) < 1.2:
            cvd_score = max(cvd_score, 0.8)  # Delta stabilization on bounce
        components["cvd_delta"] = min(1.0, cvd_score) * cls.WEIGHTS["cvd_delta"]

        # 3. Price Response & Retest (0-15)
        price_resp = 0.0
        upper_wick = absorption_data.get("upper_wick_ratio", 0.0)
        if upper_wick >= 0.35:
            price_resp += 0.6
        elif upper_wick >= 0.22:
            price_resp += 0.4
        if absorption_data.get("seller_absorption_type") == "EXHAUSTION_REVERSAL":
            price_resp += 0.4
        if is_retest and structure_m15.get("choch_bearish", False) and upper_wick >= 0.20:
            price_resp = max(price_resp, 0.8)
        components["price_response"] = min(1.0, price_resp) * cls.WEIGHTS["price_response"]

        # 4. M15 Structure CHoCH / BOS / Liquidity Sweep (0-15)
        m15_score = 0.0
        if structure_m15.get("choch_bearish", False):
            m15_score = 1.0
        elif structure_m15.get("liquidity_sweep_bearish", False):
            m15_score = 0.8  # Strong institutional rejection of higher liquidity
        elif structure_m15.get("structure_bias") == "BEARISH":
            m15_score = 0.5
        components["m15_structure"] = m15_score * cls.WEIGHTS["m15_structure"]

        # 5. H1 / H4 Confirmation (0-10)
        h1_score = 0.0
        h1_bias = structure_h1.get("structure_bias", "NEUTRAL")
        if h1_bias == "BEARISH":
            h1_score = 0.85
        elif h1_bias == "NEUTRAL":
            h1_score = 0.5
        elif h1_bias == "BULLISH":
            h1_score = 0.0  # Counter-trend penalty: no points if higher timeframe opposes

        if structure_h1.get("choch_bearish", False):
            h1_score = 1.0

        if structure_h4:
            h4_bias = structure_h4.get("structure_bias", "NEUTRAL")
            if h4_bias == "BEARISH":
                h1_score = min(1.0, h1_score + 0.15)
            elif h4_bias == "BULLISH":
                h1_score = max(0.0, h1_score - 0.25)

        components["h1_h4_confirmation"] = h1_score * cls.WEIGHTS["h1_h4_confirmation"]

        # 6. OI and Liquidations (0-10)
        deriv_score = derivatives_data.get("derivatives_score", 5.0) / 10.0
        if derivatives_data.get("short_liquidations_absorbed", False):
            deriv_score = min(1.0, deriv_score + 0.2)
        components["oi_liquidations"] = deriv_score * cls.WEIGHTS["oi_liquidations"]

        # 7. Volume, Spread, Liquidity, and Order Book Imbalance (0-5)
        rvol = absorption_data.get("rvol", 1.0)
        obi = absorption_data.get("orderbook_imbalance", 0.0)
        liq_score = 0.4
        if rvol >= 1.2 and spread_pct <= 0.03:
            liq_score = 0.8
            if obi < -0.15:  # Order Book Imbalance in favor of asks (passive absorption)
                liq_score = 1.0
        elif rvol >= 1.0 and spread_pct <= 0.04:
            liq_score = 0.6
        components["volume_liquidity"] = liq_score * cls.WEIGHTS["volume_liquidity"]

        # 8. Structural Zone (0-5) - Real proximity calculation
        if "resistance_proximity" in structure_m15:
            zone_score = float(structure_m15["resistance_proximity"])
        elif structure_m15.get("choch_bearish") or structure_m15.get("liquidity_sweep_bearish"):
            zone_score = 0.8
        else:
            zone_score = 0.3

        if absorption_data.get("vwap_confluence_bearish", False):
            zone_score = min(1.0, zone_score + 0.2)

        components["structural_zone"] = zone_score * cls.WEIGHTS["structural_zone"]

        total_score = sum(components.values())

        state = "NO_SIGNAL"
        if total_score >= 85.0 and structure_m15.get("choch_bearish", False):
            state = "CONFIRMED_REVERSAL"
        elif total_score >= 70.0:
            state = "POTENTIAL_ABSORPTION"
        elif total_score >= 55.0:
            state = "PRE_ALERT"
        elif total_score >= 40.0:
            state = "OBSERVATION"

        return round(total_score, 1), state, components

