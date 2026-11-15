"""Market Structure and Anti-Repaint CHoCH / BOS Detection.

Strict Rules:
- Evaluates exclusively on CLOSED candles.
- Confirmed pivot highs/lows using a symmetric backward lookback (no future lookahead).
- Identifies CHoCH (Change of Character) and BOS (Break of Structure).
- Computes structural support/resistance zones.
"""

from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np


class MarketStructure:
    """Anti-repainting market structure analyzer."""

    def __init__(self, pivot_left: int = 3, pivot_right: int = 2):
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right

    def find_pivots(self, df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Detects confirmed swing highs and swing lows without future lookahead.

        A pivot at index `i` is only confirmed once `i + pivot_right` is reached.
        """
        swing_highs = []
        swing_lows = []
        n = len(df)

        if n < self.pivot_left + self.pivot_right + 1:
            return swing_highs, swing_lows

        highs = df["high"].values
        lows = df["low"].values
        times = df["open_time"].values

        # Only iterate up to n - pivot_right to guarantee candle is confirmed
        for i in range(self.pivot_left, n - self.pivot_right):
            # Check swing high
            is_high = True
            for left in range(1, self.pivot_left + 1):
                if highs[i] <= highs[i - left]:
                    is_high = False
                    break
            if is_high:
                for right in range(1, self.pivot_right + 1):
                    if highs[i] < highs[i + right]:
                        is_high = False
                        break
            if is_high:
                swing_highs.append({"index": i, "open_time": times[i], "price": highs[i]})

            # Check swing low
            is_low = True
            for left in range(1, self.pivot_left + 1):
                if lows[i] >= lows[i - left]:
                    is_low = False
                    break
            if is_low:
                for right in range(1, self.pivot_right + 1):
                    if lows[i] > lows[i + right]:
                        is_low = False
                        break
            if is_low:
                swing_lows.append({"index": i, "open_time": times[i], "price": lows[i]})

        return swing_highs, swing_lows

    def evaluate_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluates latest structural state, CHoCH, liquidity sweeps, and support/resistance zones."""
        min_required = self.pivot_left + self.pivot_right + 1
        if len(df) < min_required:
            return {
                "structure_bias": "NEUTRAL",
                "last_swing_high": None,
                "last_swing_low": None,
                "choch_bullish": False,
                "choch_bearish": False,
                "choch_bars_ago": None,
                "choch_origin_low": None,
                "choch_origin_high": None,
                "choch_impulse_high": None,
                "choch_impulse_low": None,
                "choch_level": None,
                "is_choch_retest": False,
                "bos_bullish": False,
                "bos_bearish": False,
                "bos_bars_ago": None,
                "bos_origin_low": None,
                "bos_origin_high": None,
                "bos_impulse_high": None,
                "bos_impulse_low": None,
                "is_bos_retest": False,
                "liquidity_sweep_bullish": False,
                "liquidity_sweep_bearish": False,
                "support_zone": None,
                "resistance_zone": None,
                "support_proximity": 0.0,
                "resistance_proximity": 0.0,
            }

        swing_highs, swing_lows = self.find_pivots(df)

        last_high = swing_highs[-1]["price"] if swing_highs else None
        last_low = swing_lows[-1]["price"] if swing_lows else None

        last_bar = df.iloc[-1]
        last_close = last_bar["close"]
        last_high_price = last_bar["high"]
        last_low_price = last_bar["low"]
        prev_close = df["close"].iloc[-2] if len(df) > 1 else last_close

        # CHoCH Detection:
        # Evaluates breakout above recent swing high or below swing low.
        # Persists across a window of up to 8 bars as long as price holds above the move's origin swing low (Dow Theory / SMC).
        choch_bullish = False
        choch_bearish = False
        choch_bars_ago = None
        choch_origin_low = None
        choch_origin_high = None
        choch_impulse_high = None
        choch_impulse_low = None
        choch_level = None
        is_choch_retest = False

        lookback = min(8, len(df))
        if last_high:
            for k in range(1, lookback + 1):
                idx = -k
                c_close = df["close"].iloc[idx]
                p_close = df["close"].iloc[idx - 1] if abs(idx - 1) <= len(df) else c_close
                if p_close <= last_high and c_close > last_high:
                    # Breakout bar found at index idx!
                    # Find origin low: lowest low in the lookback leading into the breakout
                    origin_window = df.iloc[-min(14, len(df)):idx + 1] if idx < -1 else df.iloc[-min(14, len(df)):]
                    origin_lo = float(origin_window["low"].min()) if len(origin_window) > 0 else (last_low or last_close * 0.98)
                    if last_low and last_low < last_high:
                        origin_lo = min(origin_lo, last_low)

                    # Impulse high: highest high reached since the breakout
                    imp_hi = float(df["high"].iloc[idx:].max())

                    # A Bullish CHoCH remains valid as long as price does NOT break below the origin low
                    if last_close >= origin_lo * 0.999:
                        choch_bullish = True
                        choch_bars_ago = k - 1
                        choch_origin_low = origin_lo
                        choch_impulse_high = imp_hi
                        choch_level = last_high
                        # If price retraced below the impulse high but is holding above origin low, it's a retest!
                        if last_close < imp_hi and last_close >= origin_lo:
                            is_choch_retest = True
                        break

        if last_low and not choch_bullish:
            for k in range(1, lookback + 1):
                idx = -k
                c_close = df["close"].iloc[idx]
                p_close = df["close"].iloc[idx - 1] if abs(idx - 1) <= len(df) else c_close
                if p_close >= last_low and c_close < last_low:
                    # Breakdown bar found at index idx!
                    origin_window = df.iloc[-min(14, len(df)):idx + 1] if idx < -1 else df.iloc[-min(14, len(df)):]
                    origin_hi = float(origin_window["high"].max()) if len(origin_window) > 0 else (last_high or last_close * 1.02)
                    if last_high and last_high > last_low:
                        origin_hi = max(origin_hi, last_high)

                    # Impulse low: lowest low reached since the breakdown
                    imp_lo = float(df["low"].iloc[idx:].min())

                    # A Bearish CHoCH remains valid as long as price does NOT break above the origin high
                    if last_close <= origin_hi * 1.001:
                        choch_bearish = True
                        choch_bars_ago = k - 1
                        choch_origin_high = origin_hi
                        choch_impulse_low = imp_lo
                        choch_level = last_low
                        # If price rebounded above the impulse low but is holding below origin high, it's a retest!
                        if last_close > imp_lo and last_close <= origin_hi:
                            is_choch_retest = True
                        break

        # Institutional Liquidity Sweep Detection (Stop-hunt + Reclamation):
        # Bullish Sweep: Price pierced below swing low (grabbed liquidity), but closed back above it!
        liquidity_sweep_bullish = False
        if last_low and last_low_price < last_low and last_close >= last_low:
            liquidity_sweep_bullish = True

        # Bearish Sweep: Price pierced above swing high (grabbed liquidity), but closed back below it!
        liquidity_sweep_bearish = False
        if last_high and last_high_price > last_high and last_close <= last_high:
            liquidity_sweep_bearish = True

        # Determine prevailing market bias (CHoCH immediately flips the bias)
        bias = "NEUTRAL"
        if choch_bullish:
            bias = "BULLISH"
        elif choch_bearish:
            bias = "BEARISH"
        elif len(swing_highs) >= 2 and len(swing_lows) >= 2:
            if swing_highs[-1]["price"] > swing_highs[-2]["price"] and swing_lows[-1]["price"] > swing_lows[-2]["price"]:
                bias = "BULLISH"
            elif swing_highs[-1]["price"] < swing_highs[-2]["price"] and swing_lows[-1]["price"] < swing_lows[-2]["price"]:
                bias = "BEARISH"

        # BOS (Break of Structure) Continuation Detection:
        # After CHoCH establishes the new trend direction, subsequent breakouts of swing points
        # in the trend direction represent BOS (trend continuation waves).
        bos_bullish = False
        bos_bearish = False
        bos_bars_ago = None
        bos_origin_low = None
        bos_origin_high = None
        bos_impulse_high = None
        bos_impulse_low = None
        is_bos_retest = False

        if bias == "BULLISH" or choch_bullish:
            # Check for subsequent breakout above choch_impulse_high or prior swing high
            ref_high = choch_impulse_high or (swing_highs[-2]["price"] if len(swing_highs) >= 2 else last_high)
            if ref_high:
                for k in range(1, lookback + 1):
                    idx = -k
                    c_close = df["close"].iloc[idx]
                    p_close = df["close"].iloc[idx - 1] if abs(idx - 1) <= len(df) else c_close
                    if p_close <= ref_high and c_close > ref_high:
                        bos_bullish = True
                        bos_bars_ago = k - 1
                        bos_origin_low = choch_origin_low or (swing_lows[-1]["price"] if swing_lows else None)
                        bos_impulse_high = float(df["high"].iloc[idx:].max())
                        if last_close < bos_impulse_high and bos_origin_low and last_close >= bos_origin_low:
                            is_bos_retest = True
                        break

        if (bias == "BEARISH" or choch_bearish) and not bos_bullish:
            ref_low = choch_impulse_low or (swing_lows[-2]["price"] if len(swing_lows) >= 2 else last_low)
            if ref_low:
                for k in range(1, lookback + 1):
                    idx = -k
                    c_close = df["close"].iloc[idx]
                    p_close = df["close"].iloc[idx - 1] if abs(idx - 1) <= len(df) else c_close
                    if p_close >= ref_low and c_close < ref_low:
                        bos_bearish = True
                        bos_bars_ago = k - 1
                        bos_origin_high = choch_origin_high or (swing_highs[-1]["price"] if swing_highs else None)
                        bos_impulse_low = float(df["low"].iloc[idx:].min())
                        if last_close > bos_impulse_low and bos_origin_high and last_close <= bos_origin_high:
                            is_bos_retest = True
                        break

        # Calculate structural proximity (0.0 to 1.0)
        support_prox = 0.0
        if last_low and last_low > 0:
            dist_pct = abs(last_close - last_low) / last_low
            if dist_pct <= 0.003:  # within 0.3%
                support_prox = 1.0
            elif dist_pct <= 0.008:  # within 0.8%
                support_prox = max(0.2, 1.0 - (dist_pct / 0.008))

        resistance_prox = 0.0
        if last_high and last_high > 0:
            dist_pct = abs(last_close - last_high) / last_high
            if dist_pct <= 0.003:
                resistance_prox = 1.0
            elif dist_pct <= 0.008:
                resistance_prox = max(0.2, 1.0 - (dist_pct / 0.008))

        return {
            "structure_bias": bias,
            "last_swing_high": last_high,
            "last_swing_low": last_low,
            "choch_bullish": choch_bullish,
            "choch_bearish": choch_bearish,
            "choch_bars_ago": choch_bars_ago,
            "choch_origin_low": choch_origin_low,
            "choch_origin_high": choch_origin_high,
            "choch_impulse_high": choch_impulse_high,
            "choch_impulse_low": choch_impulse_low,
            "choch_level": choch_level,
            "is_choch_retest": is_choch_retest,
            "bos_bullish": bos_bullish,
            "bos_bearish": bos_bearish,
            "bos_bars_ago": bos_bars_ago,
            "bos_origin_low": bos_origin_low,
            "bos_origin_high": bos_origin_high,
            "bos_impulse_high": bos_impulse_high,
            "bos_impulse_low": bos_impulse_low,
            "is_bos_retest": is_bos_retest,
            "liquidity_sweep_bullish": liquidity_sweep_bullish,
            "liquidity_sweep_bearish": liquidity_sweep_bearish,
            "support_zone": (last_low * 0.998, last_low * 1.002) if last_low else None,
            "resistance_zone": (last_high * 0.998, last_high * 1.002) if last_high else None,
            "support_proximity": support_prox,
            "resistance_proximity": resistance_prox,
        }
