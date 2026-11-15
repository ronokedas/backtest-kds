"""Async WebSocket client for streaming real-time Binance market data.

Supports:
- Individual and multiplexed streams for Spot and USDⓈ-M Futures.
- Real-time `aggTrade` streams with aggressive taker side parsing.
- Real-time `bookTicker` for spread and best bid/ask.
- Real-time `depth5@100ms` for order book imbalance.
- Automatic reconnection with exponential backoff.
- Sequence validation and latency monitoring.
"""

import asyncio
import json
import logging
import time
from typing import List, Callable, Optional, Dict, Any, Awaitable
import websockets
from data.data_validator import DataValidator, AggTradeData

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """Manages resilient WebSocket connections to Binance public streams."""

    SPOT_WS_BASE = "wss://stream.binance.com:9443"
    FUTURES_WS_BASE = "wss://fstream.binance.com"

    def __init__(
        self,
        market: str = "futures_usdm",
        validator: Optional[DataValidator] = None,
        max_reconnect_attempts: int = 10,
    ):
        self.market = market
        self.validator = validator or DataValidator()
        self.max_reconnect_attempts = max_reconnect_attempts
        self.ws_base = self.FUTURES_WS_BASE if market == "futures_usdm" else self.SPOT_WS_BASE

        self._running = False
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._trade_callbacks: List[Callable[[AggTradeData], Awaitable[None]]] = []
        self._raw_callbacks: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        self.last_heartbeat_ms: int = 0
        self.total_messages_received: int = 0
        self.total_gaps_detected: int = 0

    def add_trade_callback(self, callback: Callable[[AggTradeData], Awaitable[None]]):
        """Registers an async callback for validated aggTrade events."""
        self._trade_callbacks.append(callback)

    def add_raw_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Registers an async callback for generic stream messages."""
        self._raw_callbacks.append(callback)

    def build_stream_url(self, streams: List[str]) -> str:
        """Builds multiplexed stream URL."""
        stream_path = "/".join(streams)
        return f"{self.ws_base}/stream?streams={stream_path}"

    async def start(self, symbols: List[str], stream_types: Optional[List[str]] = None):
        """Starts WebSocket listening loop for given symbols and stream types."""
        stream_types = stream_types or ["aggTrade", "bookTicker"]
        streams = []
        for s in symbols:
            s_lower = s.lower()
            for st in stream_types:
                streams.append(f"{s_lower}@{st}")

        url = self.build_stream_url(streams)
        self._running = True
        reconnect_count = 0
        backoff_sec = 1.0

        logger.info(f"Starting Binance WS for {len(streams)} streams on {self.market}...")

        while self._running and reconnect_count < self.max_reconnect_attempts:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=None,  # Binance server initiates ping; client automatically responds with pong
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    self._ws = ws
                    reconnect_count = 0
                    backoff_sec = 1.0
                    logger.info("Connected to Binance WebSocket stream successfully.")

                    while self._running:
                        message = await ws.recv()
                        self.total_messages_received += 1
                        self.last_heartbeat_ms = int(time.time() * 1000)
                        await self._handle_message(message)

            except (websockets.ConnectionClosed, asyncio.TimeoutError, Exception) as e:
                if not self._running:
                    break
                reconnect_count += 1
                logger.warning(
                    f"WebSocket disconnected ({type(e).__name__}: {e}). Reconnecting ({reconnect_count}/{self.max_reconnect_attempts}) in {backoff_sec}s..."
                )
                await asyncio.sleep(backoff_sec)
                backoff_sec = min(backoff_sec * 2.0, 30.0)

        if reconnect_count >= self.max_reconnect_attempts:
            logger.error("Max WebSocket reconnect attempts reached. Stream stopped.")

    async def _handle_message(self, raw_message: str):
        """Parses and validates incoming message."""
        try:
            data = json.loads(raw_message)
        except Exception as e:
            logger.error(f"Failed to parse JSON WS message: {e}")
            return

        payload = data.get("data", data)
        event_type = payload.get("e")
        recv_time = int(time.time() * 1000)

        # Handle aggTrade
        if event_type == "aggTrade":
            is_valid, trade, warning = self.validator.validate_agg_trade(payload, recv_time_ms=recv_time)
            if warning and "gap" in warning.lower():
                self.total_gaps_detected += 1
                logger.warning(warning)

            if is_valid and trade:
                for cb in self._trade_callbacks:
                    try:
                        await cb(trade)
                    except Exception as err:
                        logger.error(f"Error in trade callback: {err}")

        # Dispatch to raw callbacks
        for raw_cb in self._raw_callbacks:
            try:
                await raw_cb(payload)
            except Exception as err:
                logger.error(f"Error in raw callback: {err}")

    async def stop(self):
        """Stops the WebSocket stream."""
        self._running = False
        if self._ws:
            await self._ws.close()
            logger.info("Binance WebSocket client closed.")

    async def subscribe(self, symbols: List[str], stream_types: Optional[List[str]] = None):
        """Dynamically subscribes to additional symbol streams over the open WebSocket."""
        stream_types = stream_types or ["aggTrade", "bookTicker", "forceOrder"]
        streams = []
        for s in symbols:
            s_lower = s.lower()
            for st in stream_types:
                streams.append(f"{s_lower}@{st}")

        is_open = getattr(self._ws, "open", True) if self._ws else False
        if self._ws and is_open and streams:
            payload = {
                "method": "SUBSCRIBE",
                "params": streams,
                "id": int(time.time() * 1000),
            }
            try:
                await self._ws.send(json.dumps(payload))
                logger.info(f"Dynamically subscribed to streams: {streams}")
            except Exception as e:
                logger.error(f"Failed to dynamic subscribe WS: {e}")

    async def unsubscribe(self, symbols: List[str], stream_types: Optional[List[str]] = None):
        """Dynamically unsubscribes from symbol streams over the open WebSocket."""
        stream_types = stream_types or ["aggTrade", "bookTicker", "forceOrder"]
        streams = []
        for s in symbols:
            s_lower = s.lower()
            for st in stream_types:
                streams.append(f"{s_lower}@{st}")

        is_open = getattr(self._ws, "open", True) if self._ws else False
        if self._ws and is_open and streams:
            payload = {
                "method": "UNSUBSCRIBE",
                "params": streams,
                "id": int(time.time() * 1000),
            }
            try:
                await self._ws.send(json.dumps(payload))
                logger.info(f"Dynamically unsubscribed from streams: {streams}")
            except Exception as e:
                logger.error(f"Failed to dynamic unsubscribe WS: {e}")

