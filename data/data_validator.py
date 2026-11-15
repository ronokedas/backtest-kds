"""Data validator for Binance market data.

Validates:
- Clock skew between local machine and Binance servers.
- Candlestick (OHLCV) structural integrity and chronological continuity.
- Trade stream continuity, deduplication, and aggressor side classification.
"""

from typing import List, Dict, Any, Tuple, Optional
import time
from pydantic import BaseModel, Field, field_validator


class KlineData(BaseModel):
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: int
    quote_volume: float
    trades_count: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float

    @field_validator("high")
    @classmethod
    def validate_high(cls, v: float, info) -> float:
        data = info.data
        if "low" in data and v < data["low"]:
            raise ValueError(f"High ({v}) cannot be less than low ({data['low']})")
        return v


class AggTradeData(BaseModel):
    event_time: int = Field(alias="E")
    symbol: str = Field(alias="s")
    aggregate_trade_id: int = Field(alias="a")
    price: float = Field(alias="p")
    quantity: float = Field(alias="q")
    first_trade_id: int = Field(alias="f")
    last_trade_id: int = Field(alias="l")
    trade_time: int = Field(alias="T")
    is_buyer_maker: bool = Field(alias="m")

    @property
    def aggressor_side(self) -> str:
        """Determines the aggressive (taker) market participant.

        m = True: Buyer is maker -> Taker (aggressor) is SELLER.
        m = False: Buyer is taker -> Taker (aggressor) is BUYER.
        """
        return "SELL" if self.is_buyer_maker else "BUY"

    @property
    def signed_delta(self) -> float:
        """Signed delta volume (+qty for taker buy, -qty for taker sell)."""
        return -self.quantity if self.is_buyer_maker else self.quantity

    @property
    def signed_notional_delta(self) -> float:
        """Signed notional delta in USD/quote currency."""
        return self.signed_delta * self.price


class DataValidator:
    """Validates real-time and historical Binance market data integrity."""

    def __init__(self, max_clock_skew_ms: int = 2000):
        self.max_clock_skew_ms = max_clock_skew_ms
        self.last_trade_ids: Dict[str, int] = {}
        self.seen_agg_trade_ids: set = set()

    def validate_clock_skew(self, local_time_ms: int, server_time_ms: int) -> Tuple[bool, int, str]:
        """Validates if local clock is synchronized with Binance server time."""
        skew_ms = abs(local_time_ms - server_time_ms)
        is_valid = skew_ms <= self.max_clock_skew_ms
        message = (
            f"Clock synchronized (drift: {skew_ms}ms)"
            if is_valid
            else f"CRITICAL: Clock drift excessive ({skew_ms}ms > {self.max_clock_skew_ms}ms)"
        )
        return is_valid, skew_ms, message

    @staticmethod
    def parse_kline_raw(kline: list) -> KlineData:
        """Parses a raw Binance REST/WS Kline array into a strongly-typed KlineData."""
        return KlineData(
            open_time=int(kline[0]),
            open=float(kline[1]),
            high=float(kline[2]),
            low=float(kline[3]),
            close=float(kline[4]),
            volume=float(kline[5]),
            close_time=int(kline[6]),
            quote_volume=float(kline[7]),
            trades_count=int(kline[8]),
            taker_buy_base_volume=float(kline[9]),
            taker_buy_quote_volume=float(kline[10]),
        )

    def validate_kline_series(
        self, klines: List[KlineData], timeframe_ms: int
    ) -> Tuple[bool, List[str]]:
        """Validates an entire series of OHLCV candles for integrity and gaps."""
        errors: List[str] = []
        if not klines:
            return False, ["Kline series is empty"]

        for i, kline in enumerate(klines):
            # 1. Structural OHLC integrity
            if kline.high < kline.low:
                errors.append(f"Index {i} [{kline.open_time}]: High ({kline.high}) < Low ({kline.low})")
            if kline.high < max(kline.open, kline.close):
                errors.append(f"Index {i} [{kline.open_time}]: High ({kline.high}) < max(Open, Close)")
            if kline.low > min(kline.open, kline.close):
                errors.append(f"Index {i} [{kline.open_time}]: Low ({kline.low}) > min(Open, Close)")
            if kline.volume < 0:
                errors.append(f"Index {i} [{kline.open_time}]: Negative volume ({kline.volume})")
            if kline.taker_buy_base_volume > kline.volume * 1.0001:  # small float tolerance
                errors.append(
                    f"Index {i} [{kline.open_time}]: Taker buy volume ({kline.taker_buy_base_volume}) > total volume ({kline.volume})"
                )

            # 2. Chronological continuity and gaps
            if i > 0:
                prev = klines[i - 1]
                expected_open = prev.open_time + timeframe_ms
                if kline.open_time != expected_open:
                    errors.append(
                        f"Gap detected between index {i-1} ({prev.open_time}) and {i} ({kline.open_time}). Expected {expected_open}"
                    )
                if kline.open_time <= prev.open_time:
                    errors.append(
                        f"Non-monotonic timestamp at index {i}: current {kline.open_time} <= prev {prev.open_time}"
                    )

        return len(errors) == 0, errors

    def validate_agg_trade(self, raw_trade: dict, recv_time_ms: Optional[int] = None) -> Tuple[bool, Optional[AggTradeData], Optional[str]]:
        """Validates a raw Binance aggTrade message."""
        try:
            trade = AggTradeData(**raw_trade)
        except Exception as e:
            return False, None, f"Schema validation failed: {str(e)}"

        # Deduplication check
        if trade.aggregate_trade_id in self.seen_agg_trade_ids:
            return False, trade, f"Duplicate aggregate_trade_id: {trade.aggregate_trade_id}"
        
        # Keep sliding memory of trade IDs to prevent unbounded memory growth
        self.seen_agg_trade_ids.add(trade.aggregate_trade_id)
        if len(self.seen_agg_trade_ids) > 100_000:
            # prune oldest half
            self.seen_agg_trade_ids = set(list(self.seen_agg_trade_ids)[50_000:])

        # Sequence gap check per symbol
        symbol = trade.symbol
        if symbol in self.last_trade_ids:
            expected_next = self.last_trade_ids[symbol] + 1
            if trade.aggregate_trade_id > expected_next:
                warning = f"Trade sequence gap for {symbol}: expected {expected_next}, got {trade.aggregate_trade_id}"
                self.last_trade_ids[symbol] = trade.aggregate_trade_id
                return True, trade, warning
            elif trade.aggregate_trade_id < expected_next:
                # Out-of-order trade
                return True, trade, f"Out of order trade for {symbol}: expected >= {expected_next}, got {trade.aggregate_trade_id}"

        self.last_trade_ids[symbol] = trade.aggregate_trade_id

        # Latency check if recv_time_ms is provided
        if recv_time_ms is not None:
            latency_ms = recv_time_ms - trade.event_time
            if latency_ms > self.max_clock_skew_ms:
                return True, trade, f"High event latency: {latency_ms}ms"

        return True, trade, None

