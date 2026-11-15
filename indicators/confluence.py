"""Technical Confluence Indicators for QuantFlow — Multi-Timeframe Engine.

Calculates:
- RSI (Relative Strength Index) with Wilder smoothing
- Trend EMAs (EMA 20, EMA 50, EMA 200)
- ATR (Average True Range) and Volatility Squeeze Compression
- Fibonacci Retracement Levels (50%, 61.8% Golden Zone, 78.6%)
- Regular Bullish & Bearish CVD Divergences
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd


class ConfluenceAnalyzer:
    """Computes technical indicator confluences across multiple timeframes."""

    @staticmethod
    def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Computes RSI using Wilder's smoothed moving average (alpha = 1/period)."""
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's exponential smoothing
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Handle zero loss (all gains -> RSI = 100) and zero change (RSI = 50)
        rsi = pd.Series(
            np.where(avg_loss == 0, np.where(avg_gain > 0, 100.0, 50.0), rsi),
            index=series.index,
        )
        return rsi.fillna(50.0)

    @staticmethod
    def compute_emas(series: pd.Series, spans: Optional[List[int]] = None) -> Dict[str, pd.Series]:
        """Calculates exponential moving averages (EMA 20, 50, 200)."""
        spans = spans or [20, 50, 200]
        emas = {}
        for span in spans:
            emas[f"ema_{span}"] = series.ewm(span=span, adjust=False).mean()
        return emas

    @staticmethod
    def compute_atr(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Computes Average True Range (ATR) and ATR percentage of price."""
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - prev_close).abs()
        tr3 = (df["low"] - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = true_range.rolling(window=period, min_periods=1).mean()
        atr_pct = (atr / df["close"]) * 100.0
        return atr, atr_pct

    @staticmethod
    def detect_volatility_squeeze(
        atr_pct_series: pd.Series,
        lookback: int = 30,
        quantile_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """Detects volatility compression / squeeze (energy accumulating before explosive breakout)."""
        if len(atr_pct_series) < 10:
            return {"is_squeezed": False, "current_atr_pct": 0.0, "threshold_atr_pct": 0.0}

        current_atr_pct = float(atr_pct_series.iloc[-1])
        window = atr_pct_series.tail(lookback)
        threshold = float(window.quantile(quantile_threshold))

        is_squeezed = current_atr_pct <= threshold
        return {
            "is_squeezed": is_squeezed,
            "current_atr_pct": round(current_atr_pct, 3),
            "threshold_atr_pct": round(threshold, 3),
        }

    @staticmethod
    def compute_fibonacci_retracement(
        df: pd.DataFrame,
        lookback: int = 40,
        struct_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Computes Fibonacci retracement levels (50%, 61.8% Golden Zone, 78.6%) dynamically anchored to recent impulse swing."""
        if len(df) < 5:
            return {
                "swing_high": 0.0,
                "swing_low": 0.0,
                "fib_50": 0.0,
                "fib_618": 0.0,
                "fib_786": 0.0,
                "in_golden_zone": False,
                "distance_to_golden_zone_pct": 0.0,
                "is_retest_anchored": False,
            }

        swing_hi = None
        swing_lo = None
        is_anchored = False

        if struct_info:
            if struct_info.get("bos_bullish") and struct_info.get("bos_origin_low") and struct_info.get("bos_impulse_high"):
                swing_lo = float(struct_info["bos_origin_low"])
                swing_hi = float(struct_info["bos_impulse_high"])
                is_anchored = True
            elif struct_info.get("bos_bearish") and struct_info.get("bos_origin_high") and struct_info.get("bos_impulse_low"):
                swing_hi = float(struct_info["bos_origin_high"])
                swing_lo = float(struct_info["bos_impulse_low"])
                is_anchored = True
            elif struct_info.get("choch_bullish") and struct_info.get("choch_origin_low") and struct_info.get("choch_impulse_high"):
                swing_lo = float(struct_info["choch_origin_low"])
                swing_hi = float(struct_info["choch_impulse_high"])
                is_anchored = True
            elif struct_info.get("choch_bearish") and struct_info.get("choch_origin_high") and struct_info.get("choch_impulse_low"):
                swing_hi = float(struct_info["choch_origin_high"])
                swing_lo = float(struct_info["choch_impulse_low"])
                is_anchored = True
            elif struct_info.get("last_swing_high") and struct_info.get("last_swing_low"):
                hi_val = float(struct_info["last_swing_high"])
                lo_val = float(struct_info["last_swing_low"])
                if hi_val > lo_val:
                    swing_hi = hi_val
                    swing_lo = lo_val

        if swing_hi is None or swing_lo is None or swing_hi <= swing_lo:
            recent = df.tail(lookback)
            swing_hi = float(recent["high"].max())
            swing_lo = float(recent["low"].min())

        span = swing_hi - swing_lo

        if span <= 0:
            return {
                "swing_high": swing_hi,
                "swing_low": swing_lo,
                "fib_50": swing_hi,
                "fib_618": swing_hi,
                "fib_786": swing_hi,
                "in_golden_zone": False,
                "distance_to_golden_zone_pct": 0.0,
                "is_retest_anchored": is_anchored,
            }

        # Pullback levels from high to low
        fib_50 = swing_hi - (span * 0.50)
        fib_618 = swing_hi - (span * 0.618)
        fib_786 = swing_hi - (span * 0.786)

        curr_price = float(df["close"].iloc[-1])
        curr_low = float(df["low"].iloc[-1]) if "low" in df.columns else curr_price
        curr_high = float(df["high"].iloc[-1]) if "high" in df.columns else curr_price

        # Golden zone is between 50% and 61.8% retracement (or extending to 78.6%)
        # Buffer around zone (4% of span or 0.25% of price)
        buf = max(span * 0.04, curr_price * 0.0025)

        # In golden zone if close is inside, OR if candle wick tested into the zone and rejected (pinbar / hammer)
        tested_by_wick = (curr_low <= fib_50 + buf and curr_low >= fib_786 - buf) or (curr_high >= fib_618 - buf and curr_high <= fib_50 + buf)
        close_in_zone = (fib_786 - buf <= curr_price <= fib_50 + buf)
        in_golden_zone = bool(close_in_zone or tested_by_wick)

        # Proximity to 50%-61.8% zone
        mid_fib = (fib_50 + fib_618) / 2.0
        ref_price = curr_low if abs(curr_low - mid_fib) < abs(curr_price - mid_fib) else curr_price
        dist_pct = 0.0 if in_golden_zone else (abs(ref_price - mid_fib) / mid_fib * 100.0)

        digits = 2 if swing_hi >= 1.0 else 6

        return {
            "swing_high": round(swing_hi, digits),
            "swing_low": round(swing_lo, digits),
            "fib_50": round(fib_50, digits),
            "fib_618": round(fib_618, digits),
            "fib_786": round(fib_786, digits),
            "in_golden_zone": in_golden_zone,
            "distance_to_golden_zone_pct": round(dist_pct, 2),
            "is_retest_anchored": is_anchored,
        }

    @staticmethod
    def detect_cvd_divergence(df: pd.DataFrame, lookback: int = 10) -> Dict[str, bool]:
        """Detects regular Bullish and Bearish CVD divergences."""
        if len(df) < lookback + 2 or "cvd" not in df.columns:
            return {"bullish_divergence": False, "bearish_divergence": False}

        current_low = float(df["low"].iloc[-1])
        prior_low = float(df["low"].iloc[-lookback:-1].min())
        current_cvd = float(df["cvd"].iloc[-1])
        prior_cvd = float(df["cvd"].iloc[-lookback:-1].min())

        current_high = float(df["high"].iloc[-1])
        prior_high = float(df["high"].iloc[-lookback:-1].max())
        prior_cvd_high = float(df["cvd"].iloc[-lookback:-1].max())

        # Bullish: Lower price low, but CVD forms a higher low (aggressive sellers absorbed)
        bullish_div = (current_low <= prior_low) and (current_cvd > prior_cvd)

        # Bearish: Higher price high, but CVD forms a lower high (aggressive buyers absorbed)
        bearish_div = (current_high >= prior_high) and (current_cvd < prior_cvd_high)

        return {
            "bullish_divergence": bullish_div,
            "bearish_divergence": bearish_div,
        }

    @staticmethod
    def detect_retest_cvd_divergence(
        df: pd.DataFrame,
        struct_info: Optional[Dict[str, Any]] = None,
        lookback: int = 24,
    ) -> Dict[str, Any]:
        """Detects institutional absorption divergence during structural pullbacks (Retest CVD Divergence).
        
        Classic SMC / Order Flow Pattern:
        - Price made an impulse high (CHoCH / BOS), and is now pulling back (making lower highs and lower lows on M15).
        - However, the cumulative volume delta (CVD) holds higher lows or flat/positive slope.
        - Meaning: Aggressive selling by retail is meeting deep passive limit Bids without driving order flow negative.
        """
        if len(df) < 8 or "cvd" not in df.columns:
            return {
                "bullish_absorption_divergence": False,
                "bearish_absorption_divergence": False,
                "price_drop_pct": 0.0,
                "price_rise_pct": 0.0,
            }

        curr_close = float(df["close"].iloc[-1])
        curr_cvd = float(df["cvd"].iloc[-1])

        window = df.tail(lookback)
        peak_idx = window["high"].idxmax()
        peak_price = float(window.loc[peak_idx, "high"])
        peak_cvd = float(window.loc[peak_idx, "cvd"])

        trough_idx = window["low"].idxmin()
        trough_price = float(window.loc[trough_idx, "low"])
        trough_cvd = float(window.loc[trough_idx, "cvd"])

        # Bullish Pullback Divergence:
        # Price made significant retracement from peak (e.g. >= 0.4% or >= 38.2% of impulse),
        # but CVD held shallow retracement (CVD Retrace Ratio <= 0.40) or formed higher lows
        price_drop_pct = (peak_price - curr_close) / peak_price * 100.0 if peak_price > 0 else 0.0
        bullish_abs = False

        if peak_idx != window.index[-1] and price_drop_pct >= 0.4:
            retest_window = window.loc[peak_idx:]
            if len(retest_window) >= 2:
                retest_min_cvd = float(retest_window["cvd"].min())
                cvd_impulse_rise = peak_cvd - trough_cvd
                price_impulse_rise = peak_price - trough_price

                if cvd_impulse_rise > 0 and price_impulse_rise > 0:
                    cvd_retrace = peak_cvd - retest_min_cvd
                    cvd_retrace_ratio = cvd_retrace / cvd_impulse_rise
                    price_retrace_ratio = (peak_price - curr_close) / price_impulse_rise

                    # Retest Divergence: Price retraced heavily while CVD barely retraced
                    if cvd_retrace_ratio <= 0.40 and price_retrace_ratio >= 0.35:
                        bullish_abs = True
                    elif retest_min_cvd > trough_cvd and cvd_retrace_ratio <= 0.50:
                        bullish_abs = True
                elif curr_cvd > trough_cvd:
                    bullish_abs = True

        # Bearish Bounce Divergence:
        # Price made significant bounce into resistance, but CVD barely rose (Ask Absorption)
        price_rise_pct = (curr_close - trough_price) / trough_price * 100.0 if trough_price > 0 else 0.0
        bearish_abs = False

        if trough_idx != window.index[-1] and price_rise_pct >= 0.4:
            bounce_window = window.loc[trough_idx:]
            if len(bounce_window) >= 2:
                bounce_max_cvd = float(bounce_window["cvd"].max())
                cvd_impulse_drop = trough_cvd - peak_cvd
                price_impulse_drop = peak_price - trough_price

                if cvd_impulse_drop < 0 and price_impulse_drop > 0:
                    cvd_bounce = bounce_max_cvd - trough_cvd
                    cvd_bounce_ratio = cvd_bounce / abs(cvd_impulse_drop)
                    price_bounce_ratio = (curr_close - trough_price) / price_impulse_drop

                    if cvd_bounce_ratio <= 0.40 and price_bounce_ratio >= 0.35:
                        bearish_abs = True
                    elif bounce_max_cvd < peak_cvd and cvd_bounce_ratio <= 0.50:
                        bearish_abs = True
                elif curr_cvd < peak_cvd:
                    bearish_abs = True
            bounce_window = window.loc[trough_idx:]
            if len(bounce_window) >= 2:
                bounce_max_cvd = float(bounce_window["cvd"].max())
                cvd_impulse_drop = trough_cvd - peak_cvd
                if cvd_impulse_drop > 0:
                    cvd_bounce = bounce_max_cvd - trough_cvd
                    cvd_bounce_ratio = cvd_bounce / cvd_impulse_drop
                    if cvd_bounce_ratio <= 0.45 or bounce_max_cvd < peak_cvd:
                        bearish_abs = True
                elif curr_cvd < peak_cvd:
                    bearish_abs = True

        return {
            "bullish_absorption_divergence": bullish_abs,
            "bearish_absorption_divergence": bearish_abs,
            "price_drop_pct": round(price_drop_pct, 2),
            "price_rise_pct": round(price_rise_pct, 2),
        }

    @staticmethod
    def detect_m5_tactical_trigger(m5_df: pd.DataFrame) -> Dict[str, Any]:
        """Detects M5 Tactical Trigger (Bullish/Bearish Engulfing / Structure breakout) to confirm entries and filter M1 noise."""
        if len(m5_df) < 3:
            return {
                "bullish_engulfing": False,
                "bearish_engulfing": False,
                "m5_aligned_long": False,
                "m5_aligned_short": False,
                "m5_delta": 0.0,
            }

        curr = m5_df.iloc[-1]
        prev = m5_df.iloc[-2]

        curr_open = float(curr["open"])
        curr_close = float(curr["close"])
        curr_high = float(curr["high"])
        curr_low = float(curr["low"])
        prev_open = float(prev["open"])
        prev_close = float(prev["close"])
        prev_high = float(prev["high"])
        prev_low = float(prev["low"])

        is_prev_bearish = prev_close <= prev_open
        is_curr_bullish = curr_close > curr_open

        # M5 Bullish Engulfing / High Breakout:
        # Green candle closes above previous candle's high, or closes above prev open with higher high
        bullish_engulfing = is_curr_bullish and (
            (is_prev_bearish and curr_close >= prev_high) or
            (is_prev_bearish and curr_close > prev_open and curr_high > prev_high)
        )

        is_prev_bullish = prev_close >= prev_open
        is_curr_bearish = curr_close < curr_open
        bearish_engulfing = is_curr_bearish and (
            (is_prev_bullish and curr_close <= prev_low) or
            (is_prev_bullish and curr_close < prev_open and curr_low < prev_low)
        )

        m5_delta = float(curr.get("delta", 0.0))
        m5_cvd_slope = float(curr.get("cvd_slope", 0.0))

        m5_aligned_long = bool(bullish_engulfing or (is_curr_bullish and (m5_delta > 0 or m5_cvd_slope >= 0)))
        m5_aligned_short = bool(bearish_engulfing or (is_curr_bearish and (m5_delta < 0 or m5_cvd_slope <= 0)))

        return {
            "bullish_engulfing": bullish_engulfing,
            "bearish_engulfing": bearish_engulfing,
            "m5_aligned_long": m5_aligned_long,
            "m5_aligned_short": m5_aligned_short,
            "m5_delta": round(m5_delta, 2),
        }

    @classmethod
    def enrich_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Enriches dataframe with all technical confluence indicators."""
        df = df.copy()

        # 1. RSI
        df["rsi"] = cls.compute_rsi(df["close"], period=14)

        # 2. EMAs
        emas = cls.compute_emas(df["close"], spans=[20, 50, 200])
        for name, series in emas.items():
            df[name] = series

        # 3. ATR & Volatility Squeeze
        atr, atr_pct = cls.compute_atr(df, period=14)
        df["atr"] = atr
        df["atr_pct"] = atr_pct

        # 4. Volume Moving Average & RVOL
        vol_ma = df["volume"].rolling(30, min_periods=5).mean()
        df["volume_ma"] = vol_ma
        df["rvol"] = df["volume"] / vol_ma.replace(0, np.nan)
        df["rvol"] = df["rvol"].fillna(1.0)

        # 5. Candle Close Location (0.0 to 1.0)
        c_range = df["high"] - df["low"]
        df["close_location"] = (df["close"] - df["low"]) / c_range.replace(0, np.nan)
        df["close_location"] = df["close_location"].fillna(0.5)

        # 6. CVD Slope (5-bar momentum)
        if "cvd" in df.columns:
            df["cvd_slope"] = df["cvd"].diff(5).fillna(0.0)

        return df
