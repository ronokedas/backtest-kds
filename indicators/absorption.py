"""Microstructure Absorption Engine.

The core differentiator between a trivial CVD divergence and TRUE ABSORPTION.
True absorption occurs strictly when:
1. Massive aggressive taker flow (Extreme Delta spike / high z-score).
2. The price CEASES to respond proportionally to this aggressive volume (Inefficient price impact).
3. Passive limit orders (iceberg / resting liquidity) absorb all market flow.
4. Downward / upward progress stalls and price bounces or leaves rejection wicks.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


class AbsorptionMetrics:
    """Quantitative metrics measuring aggression vs price displacement."""

    def __init__(
        self,
        zscore_delta_threshold: float = 1.8,
        rvol_min: float = 1.2,
        lower_wick_rejection_min: float = 0.35,
        upper_wick_rejection_min: float = 0.35,
    ):
        self.zscore_delta_threshold = zscore_delta_threshold
        self.rvol_min = rvol_min
        self.lower_wick_rejection_min = lower_wick_rejection_min
        self.upper_wick_rejection_min = upper_wick_rejection_min

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches DataFrame with price progress, true range, and absorption ratios."""
        if len(df) < 5:
            return df

        df = df.copy()

        # 1. Price displacement and candle anatomy
        df["candle_body"] = df["close"] - df["open"]
        df["candle_range"] = (df["high"] - df["low"]).replace(0, 1e-6)
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        df["upper_wick_ratio"] = df["upper_wick"] / df["candle_range"]
        df["lower_wick_ratio"] = df["lower_wick"] / df["candle_range"]

        # 2. Relative Volume (RVOL)
        rolling_vol = df["volume"].rolling(window=20, min_periods=5).mean().replace(0, 1e-6)
        df["rvol"] = df["volume"] / rolling_vol

        # 3. Price Impact per Delta (Eff = ΔPrice / Delta)
        # Low price impact with high volume delta signifies passive absorption
        df["price_impact_per_delta"] = df["candle_body"] / df["delta"].replace(0, 1e-6)

        # 4. Absorption Ratio = |Delta| / Candle Range
        df["absorption_density"] = df["delta"].abs() / df["candle_range"]

        return df

    def evaluate_bar_absorption(self, row: pd.Series, prev_rows: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates whether a single candle exhibits TRUE absorption rather than quiet divergence.

        Returns detailed metrics and status.
        """
        delta = row.get("delta", 0.0)
        delta_zscore = row.get("delta_zscore", 0.0)
        rvol = row.get("rvol", 1.0)
        body = row.get("candle_body", 0.0)
        c_range = row.get("candle_range", 1e-6)
        lower_wick_ratio = row.get("lower_wick_ratio", 0.0)
        upper_wick_ratio = row.get("upper_wick_ratio", 0.0)

        # ---------------------------------------------------------------------
        # 1. BUYER ABSORPTION (Bids holding against heavy aggressive selling)
        # ---------------------------------------------------------------------
        # Conditions:
        # A) Heavy aggressive selling: delta_zscore <= -threshold OR extreme negative delta with rvol
        is_heavy_selling = (delta_zscore <= -self.zscore_delta_threshold) or (
            row.get("delta_ratio", 0.0) <= -0.35 and rvol >= self.rvol_min
        )

        # B) Price failure to progress downwards:
        # - Candle closed green despite negative delta (extreme absorption)
        # - OR long lower wick (rejection: sellers tried to push, got absorbed, price recovered)
        # - OR tiny negative body despite massive volume (sellers slammed the wall)
        downward_failure = False
        absorption_type_buy = None

        if is_heavy_selling:
            if body > 0:  # Green candle with heavy selling
                downward_failure = True
                absorption_type_buy = "EXHAUSTION_REVERSAL"  # Green bar on net sell delta
            elif lower_wick_ratio >= self.lower_wick_rejection_min:
                downward_failure = True
                absorption_type_buy = "LOWER_WICK_ABSORPTION"
            elif abs(body) / c_range <= 0.30:  # Doji/compressed bar on massive selling
                downward_failure = True
                absorption_type_buy = "COMPRESSION_WALL"

        # Check for multi-bar retest of the absorption zone in the last 1-4 bars
        buyer_retest = False
        seller_retest = False
        if prev_rows is not None and len(prev_rows) >= 1:
            recent_prev = prev_rows.tail(4)
            curr_low = row.get("low", 0.0)
            curr_high = row.get("high", 0.0)
            
            if curr_low > 0:
                for _, p_bar in recent_prev.iterrows():
                    p_low = p_bar.get("low", 0.0)
                    if p_low > 0 and abs(p_low - curr_low) / curr_low <= 0.0035:
                        buyer_retest = True
                        break

            if curr_high > 0:
                for _, p_bar in recent_prev.iterrows():
                    p_high = p_bar.get("high", 0.0)
                    if p_high > 0 and abs(p_high - curr_high) / curr_high <= 0.0035:
                        seller_retest = True
                        break

        buyer_absorption_score = 0.0
        if is_heavy_selling and downward_failure:
            # Score formula: combines delta magnitude, rejection quality, and RVOL
            retest_bonus = 10.0 if buyer_retest else 0.0
            intensity = min(100.0, (abs(delta_zscore) * 25.0) + (lower_wick_ratio * 30.0) + (rvol * 15.0) + retest_bonus)
            buyer_absorption_score = max(55.0, intensity)

        # ---------------------------------------------------------------------
        # 2. SELLER ABSORPTION (Asks holding against heavy aggressive buying)
        # ---------------------------------------------------------------------
        # Conditions:
        # A) Heavy aggressive buying: delta_zscore >= threshold OR extreme positive delta with rvol
        is_heavy_buying = (delta_zscore >= self.zscore_delta_threshold) or (
            row.get("delta_ratio", 0.0) >= 0.35 and rvol >= self.rvol_min
        )

        upward_failure = False
        absorption_type_sell = None

        if is_heavy_buying:
            if body < 0:  # Red candle with heavy buying
                upward_failure = True
                absorption_type_sell = "EXHAUSTION_REVERSAL"
            elif upper_wick_ratio >= self.upper_wick_rejection_min:
                upward_failure = True
                absorption_type_sell = "UPPER_WICK_ABSORPTION"
            elif abs(body) / c_range <= 0.30:  # Doji/compressed bar on massive buying
                upward_failure = True
                absorption_type_sell = "COMPRESSION_WALL"

        seller_absorption_score = 0.0
        if is_heavy_buying and upward_failure:
            retest_bonus = 10.0 if seller_retest else 0.0
            intensity = min(100.0, (delta_zscore * 25.0) + (upper_wick_ratio * 30.0) + (rvol * 15.0) + retest_bonus)
            seller_absorption_score = max(55.0, intensity)

        # VWAP Confluence Check
        vwap = row.get("vwap", 0.0)
        vwap_confluence_bullish = False
        vwap_confluence_bearish = False
        if vwap > 0:
            c = row.get("close", 0.0)
            l = row.get("low", 0.0)
            h = row.get("high", 0.0)
            if (l <= vwap and c >= vwap) or (abs(c - vwap) / vwap <= 0.003):
                vwap_confluence_bullish = True
            if (h >= vwap and c <= vwap) or (abs(c - vwap) / vwap <= 0.003):
                vwap_confluence_bearish = True

        return {
            "is_buyer_absorption": buyer_absorption_score >= 55.0,
            "buyer_absorption_score": buyer_absorption_score,
            "buyer_absorption_type": absorption_type_buy,
            "buyer_retest": buyer_retest,
            "is_seller_absorption": seller_absorption_score >= 55.0,
            "seller_absorption_score": seller_absorption_score,
            "seller_absorption_type": absorption_type_sell,
            "seller_retest": seller_retest,
            "delta_zscore": delta_zscore,
            "rvol": rvol,
            "lower_wick_ratio": lower_wick_ratio,
            "upper_wick_ratio": upper_wick_ratio,
            "vwap": vwap,
            "vwap_confluence_bullish": vwap_confluence_bullish,
            "vwap_confluence_bearish": vwap_confluence_bearish,
        }

