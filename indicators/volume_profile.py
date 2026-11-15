"""Volume Profile and Liquidity Pool Analyzer.

Calculates:
- Point of Control (POC): Price level with maximum volume transacted
- Value Area High (VAH) & Value Area Low (VAL): 70% volume distribution range
- Liquidity Sweep Detection: Price piercing VAL/VAH to trap breakout retail,
  followed by immediate reclaim and rejection
- Magnet Proximity: Distance to POC for optimal target placement
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


class VolumeProfileAnalyzer:
    """Calculates session and rolling Volume Profile with Value Area and Liquidity Sweeps."""

    @staticmethod
    def compute_volume_profile(
        df: pd.DataFrame,
        lookback: int = 150,
        rows: int = 24,
        va_percent: float = 70.0,
    ) -> Dict[str, Any]:
        """Calculates Volume Profile, POC, VAH, and VAL from historical klines."""
        if len(df) < 5:
            return {
                "poc": 0.0,
                "vah": 0.0,
                "val": 0.0,
                "range_high": 0.0,
                "range_low": 0.0,
                "total_volume": 0.0,
                "in_value_area": False,
                "val_sweep_bullish": False,
                "vah_sweep_bearish": False,
                "distance_to_poc_pct": 0.0,
                "profile_rows": [],
            }

        window = df.tail(lookback)
        p_hi = float(window["high"].max())
        p_lo = float(window["low"].min())
        span = p_hi - p_lo

        if span <= 0:
            last_p = float(df["close"].iloc[-1])
            return {
                "poc": last_p,
                "vah": last_p,
                "val": last_p,
                "range_high": p_hi,
                "range_low": p_lo,
                "total_volume": float(window["volume"].sum()),
                "in_value_area": True,
                "val_sweep_bullish": False,
                "vah_sweep_bearish": False,
                "distance_to_poc_pct": 0.0,
                "profile_rows": [],
            }

        step = span / rows
        vols = np.zeros(rows, dtype=np.float64)

        highs = window["high"].values
        lows = window["low"].values
        volumes = window["volume"].values

        for b_lo, b_hi, v in zip(lows, highs, volumes):
            if v <= 0:
                continue
            r0 = max(0, int(np.floor((b_lo - p_lo) / step)))
            r1 = min(rows - 1, int(np.floor((b_hi - p_lo) / step)))
            n_r = max(1, r1 - r0 + 1)
            v_per_row = v / n_r
            vols[r0 : r1 + 1] += v_per_row

        total_vol = float(vols.sum())
        poc_idx = int(np.argmax(vols))
        poc_price = p_lo + (poc_idx + 0.5) * step

        # Expand Value Area symmetrically from POC
        up_idx = poc_idx
        dn_idx = poc_idx
        va_sum = vols[poc_idx]
        target_vol = total_vol * (va_percent / 100.0)

        while va_sum < target_vol and (up_idx < rows - 1 or dn_idx > 0):
            up_v = vols[up_idx + 1] if up_idx < rows - 1 else -1.0
            dn_v = vols[dn_idx - 1] if dn_idx > 0 else -1.0
            if up_v >= dn_v:
                up_idx += 1
                va_sum += up_v
            else:
                dn_idx -= 1
                va_sum += dn_v

        vah_price = p_lo + (up_idx + 1) * step
        val_price = p_lo + dn_idx * step

        # Current candle liquidity sweeps
        curr_bar = df.iloc[-1]
        curr_c = float(curr_bar["close"])
        curr_l = float(curr_bar["low"])
        curr_h = float(curr_bar["high"])

        in_va = val_price <= curr_c <= vah_price

        # VAL Liquidity Sweep (Bullish Reversal):
        # Price pierced below VAL (sweeping stops/liquidity), but closed back inside Value Area!
        val_sweep = (curr_l < val_price) and (curr_c >= val_price)

        # VAH Liquidity Sweep (Bearish Reversal):
        # Price pierced above VAH (sweeping stops/liquidity), but closed back inside Value Area!
        vah_sweep = (curr_h > vah_price) and (curr_c <= vah_price)

        dist_poc = abs(curr_c - poc_price) / poc_price * 100.0 if poc_price > 0 else 0.0
        digits = 2 if p_hi >= 1.0 else 6

        return {
            "poc": round(poc_price, digits),
            "vah": round(vah_price, digits),
            "val": round(val_price, digits),
            "range_high": round(p_hi, digits),
            "range_low": round(p_lo, digits),
            "total_volume": round(total_vol, 2),
            "in_value_area": in_va,
            "val_sweep_bullish": val_sweep,
            "vah_sweep_bearish": vah_sweep,
            "distance_to_poc_pct": round(dist_poc, 2),
            "poc_idx": poc_idx,
            "up_idx": up_idx,
            "dn_idx": dn_idx,
        }

