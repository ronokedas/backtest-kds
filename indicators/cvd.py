"""Cumulative Volume Delta (CVD) and Delta Aggression Engine.

Calculates:
- Real-time tick delta from AggTrade (m field classification).
- Bar delta from candlestick data (taker buy base volume vs sell volume).
- Rolling Cumulative Volume Delta (CVD).
- Normalized Delta Z-Score to isolate statistically significant aggression spikes.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from data.data_validator import KlineData, AggTradeData


class CVDEngine:
    """Computes exact delta metrics and Cumulative Volume Delta (CVD)."""

    def __init__(self, rolling_window: int = 20):
        self.rolling_window = rolling_window
        self.cum_delta: float = 0.0

    @staticmethod
    def calculate_kline_delta(kline: KlineData) -> Tuple[float, float, float]:
        """Calculates volume metrics for a single closed or in-progress kline.

        Returns: (buy_volume, sell_volume, net_delta)
        """
        buy_volume = kline.taker_buy_base_volume
        sell_volume = max(0.0, kline.volume - buy_volume)
        net_delta = buy_volume - sell_volume
        return buy_volume, sell_volume, net_delta

    def compute_kline_series_cvd(self, klines: List[KlineData]) -> pd.DataFrame:
        """Processes a list of KlineData into a pandas DataFrame with CVD and Delta metrics."""
        if not klines:
            return pd.DataFrame()

        records = []
        running_cvd = 0.0

        for k in klines:
            buy_vol, sell_vol, delta = self.calculate_kline_delta(k)
            running_cvd += delta
            delta_ratio = (delta / k.volume) if k.volume > 0 else 0.0

            records.append(
                {
                    "open_time": k.open_time,
                    "open": k.open,
                    "high": k.high,
                    "low": k.low,
                    "close": k.close,
                    "volume": k.volume,
                    "taker_buy_volume": buy_vol,
                    "taker_sell_volume": sell_vol,
                    "delta": delta,
                    "delta_ratio": delta_ratio,
                    "cvd": running_cvd,
                }
            )

        df = pd.DataFrame(records)

        # Rolling statistics for statistical normalization (Z-Score)
        rolling_mean = df["delta"].rolling(window=self.rolling_window, min_periods=5).mean()
        rolling_std = df["delta"].rolling(window=self.rolling_window, min_periods=5).std().replace(0, 1e-6)
        df["delta_zscore"] = (df["delta"] - rolling_mean) / rolling_std
        df["delta_zscore"] = df["delta_zscore"].fillna(0.0)

        # CVD Slope (momentum)
        df["cvd_change_3"] = df["cvd"].diff(3).fillna(0.0)
        df["cvd_change_5"] = df["cvd"].diff(5).fillna(0.0)

        # VWAP calculation
        typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
        cum_vol_price = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum().replace(0, 1e-6)
        df["vwap"] = cum_vol_price / cum_vol

        return df


class RealtimeDeltaTracker:
    """Tracks real-time tick-by-tick aggression and delta from WebSocket aggTrades."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.current_bar_open_time: int = 0
        self.current_bar_delta: float = 0.0
        self.current_bar_buy_volume: float = 0.0
        self.current_bar_sell_volume: float = 0.0
        self.total_trades_count: int = 0
        self.last_price: float = 0.0
        self.running_cvd: float = 0.0
        self.recent_ticks_delta: List[float] = []

    def on_trade(self, trade: AggTradeData):
        self.last_price = trade.price
        self.total_trades_count += 1
        delta = trade.signed_delta
        self.current_bar_delta += delta
        self.running_cvd += delta

        if trade.is_buyer_maker:
            self.current_bar_sell_volume += trade.quantity
        else:
            self.current_bar_buy_volume += trade.quantity

        self.recent_ticks_delta.append(delta)
        if len(self.recent_ticks_delta) > 500:
            self.recent_ticks_delta = self.recent_ticks_delta[-250:]

    def reset_bar(self, open_time: int, initial_cvd: Optional[float] = None):
        self.current_bar_open_time = open_time
        self.current_bar_delta = 0.0
        self.current_bar_buy_volume = 0.0
        self.current_bar_sell_volume = 0.0
        if initial_cvd is not None:
            self.running_cvd = initial_cvd

    @staticmethod
    def detect_raw_divergence(df: pd.DataFrame, lookback: int = 5) -> Optional[str]:
        """Identifies simple price vs CVD divergence (NOT sufficient alone for absorption).

        Returns:
            "BULLISH_DIVERGENCE": Lower price low, but higher CVD low (weak clue).
            "BEARISH_DIVERGENCE": Higher price high, but lower CVD high (weak clue).
            None: No divergence.
        """
        if len(df) < lookback + 2:
            return None

        recent_slice = df.iloc[-lookback:]
        p_curr = recent_slice["close"].iloc[-1]
        p_min = recent_slice["low"].min()
        cvd_curr = recent_slice["cvd"].iloc[-1]
        cvd_min = recent_slice["cvd"].min()

        # Bullish divergence check
        if p_curr <= p_min and cvd_curr > cvd_min:
            return "BULLISH_DIVERGENCE"

        # Bearish divergence check
        p_max = recent_slice["high"].max()
        cvd_max = recent_slice["cvd"].max()
        if p_curr >= p_max and cvd_curr < cvd_max:
            return "BEARISH_DIVERGENCE"

        return None

