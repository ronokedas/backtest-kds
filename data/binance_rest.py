"""Async REST client for Binance Spot and USDⓈ-M Futures.

Provides access to official endpoints for market data, historical klines,
order book snapshots, and derivatives metrics with rate limit handling,
exponential backoff, and clock drift verification.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import aiohttp
from data.data_validator import DataValidator, KlineData

logger = logging.getLogger(__name__)


class BinanceRestClient:
    """Official Binance REST API client supporting Spot and USDⓈ-M Futures."""

    SPOT_URL = "https://api.binance.com"
    FUTURES_URL = "https://fapi.binance.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout_seconds: int = 10,
        max_retries: int = 3,
        validator: Optional[DataValidator] = None,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.max_retries = max_retries
        self.validator = validator or DataValidator()
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["X-MBX-APIKEY"] = self.api_key
            self._session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)
        return self._session

    async def close(self):
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Executes an HTTP request with exponential backoff on rate limits or errors."""
        session = await self._get_session()
        retries = 0
        backoff = 1.0

        while retries <= self.max_retries:
            try:
                async with session.request(method, url, params=params) as resp:
                    used_weight = resp.headers.get("x-mbx-used-weight-1m", "unknown")

                    if resp.status == 200:
                        return await resp.json()

                    if resp.status in (429, 418):
                        retry_after = int(resp.headers.get("Retry-After", int(backoff)))
                        logger.warning(
                            f"Binance rate limit hit ({resp.status}). Used weight: {used_weight}. Waiting {retry_after}s..."
                        )
                        await asyncio.sleep(retry_after)
                        retries += 1
                        backoff *= 2.0
                        continue

                    error_text = await resp.text()
                    logger.error(f"Binance API error {resp.status} on {url}: {error_text}")
                    if resp.status >= 500:
                        retries += 1
                        await asyncio.sleep(backoff)
                        backoff *= 2.0
                        continue

                    raise RuntimeError(f"Binance API error {resp.status}: {error_text}")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                retries += 1
                if retries > self.max_retries:
                    raise RuntimeError(f"Max retries exceeded for {url}: {str(e)}") from e
                logger.warning(f"Network error on {url}: {e}. Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
                backoff *= 2.0

        raise RuntimeError(f"Request failed after {self.max_retries} attempts.")

    # ------------------------------------------------------------------------
    # Time and Clock Synchronization
    # ------------------------------------------------------------------------

    async def get_server_time(self, market: str = "futures_usdm") -> int:
        """Fetches Binance server time in milliseconds."""
        base_url = self.FUTURES_URL if market == "futures_usdm" else self.SPOT_URL
        endpoint = "/fapi/v1/time" if market == "futures_usdm" else "/api/v3/time"
        data = await self._request("GET", f"{base_url}{endpoint}")
        return data["serverTime"]

    async def check_clock_skew(self, market: str = "futures_usdm") -> Tuple[bool, int, str]:
        """Checks synchronization between local system time and Binance server time."""
        t_local_start = int(time.time() * 1000)
        server_time = await self.get_server_time(market=market)
        t_local_end = int(time.time() * 1000)
        round_trip_estimate = (t_local_start + t_local_end) // 2
        return self.validator.validate_clock_skew(round_trip_estimate, server_time)

    # ------------------------------------------------------------------------
    # Exchange Information & Universe Discovery
    # ------------------------------------------------------------------------

    async def get_exchange_info(self, market: str = "futures_usdm") -> Dict[str, Any]:
        """Fetches active symbols, status, filters, and contract specifications."""
        base_url = self.FUTURES_URL if market == "futures_usdm" else self.SPOT_URL
        endpoint = "/fapi/v1/exchangeInfo" if market == "futures_usdm" else "/api/v3/exchangeInfo"
        return await self._request("GET", f"{base_url}{endpoint}")

    async def get_24hr_ticker(
        self, symbol: Optional[str] = None, market: str = "futures_usdm"
    ) -> Any:
        """Fetches 24-hour price change and rolling volume statistics."""
        base_url = self.FUTURES_URL if market == "futures_usdm" else self.SPOT_URL
        endpoint = "/fapi/v1/ticker/24hr" if market == "futures_usdm" else "/api/v3/ticker/24hr"
        params = {"symbol": symbol} if symbol else {}
        return await self._request("GET", f"{base_url}{endpoint}", params=params)

    # ------------------------------------------------------------------------
    # Historical Klines (Candlesticks) with Taker Volume
    # ------------------------------------------------------------------------

    async def get_klines(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        market: str = "futures_usdm",
    ) -> List[KlineData]:
        """Fetches historical klines with taker buy base asset volume."""
        base_url = self.FUTURES_URL if market == "futures_usdm" else self.SPOT_URL
        endpoint = "/fapi/v1/klines" if market == "futures_usdm" else "/api/v3/klines"
        params: Dict[str, Any] = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        raw_klines = await self._request("GET", f"{base_url}{endpoint}", params=params)
        return [self.validator.parse_kline_raw(k) for k in raw_klines]

    # ------------------------------------------------------------------------
    # Order Book & Book Ticker
    # ------------------------------------------------------------------------

    async def get_order_book(
        self, symbol: str, limit: int = 20, market: str = "futures_usdm"
    ) -> Dict[str, Any]:
        """Fetches snapshot of the bid/ask order book depth."""
        base_url = self.FUTURES_URL if market == "futures_usdm" else self.SPOT_URL
        endpoint = "/fapi/v1/depth" if market == "futures_usdm" else "/api/v3/depth"
        params = {"symbol": symbol, "limit": limit}
        return await self._request("GET", f"{base_url}{endpoint}", params=params)

    async def get_book_ticker(
        self, symbol: str, market: str = "futures_usdm"
    ) -> Dict[str, Any]:
        """Fetches best bid and ask price and quantity for spread calculations."""
        base_url = self.FUTURES_URL if market == "futures_usdm" else self.SPOT_URL
        endpoint = "/fapi/v1/ticker/bookTicker" if market == "futures_usdm" else "/api/v3/ticker/bookTicker"
        params = {"symbol": symbol}
        return await self._request("GET", f"{base_url}{endpoint}", params=params)

    # ------------------------------------------------------------------------
    # Futures Specific Metrics (Open Interest, Taker Ratio, Funding)
    # ------------------------------------------------------------------------

    async def get_open_interest(self, symbol: str) -> Dict[str, Any]:
        """Fetches latest Open Interest for a contract (USDⓈ-M Futures)."""
        url = f"{self.FUTURES_URL}/fapi/v1/openInterest"
        return await self._request("GET", url, params={"symbol": symbol})

    async def get_open_interest_hist(
        self, symbol: str, period: str = "15m", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetches historical Open Interest statistics."""
        url = f"{self.FUTURES_URL}/futures/data/openInterestHist"
        params = {"symbol": symbol, "period": period, "limit": limit}
        return await self._request("GET", url, params=params)

    async def get_taker_long_short_ratio(
        self, symbol: str, period: str = "15m", limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetches taker buy/sell volume and taker long/short ratio."""
        url = f"{self.FUTURES_URL}/futures/data/takerlongshortRatio"
        params = {"symbol": symbol, "period": period, "limit": limit}
        return await self._request("GET", url, params=params)

    async def get_funding_rate(
        self, symbol: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Fetches latest funding rate history."""
        url = f"{self.FUTURES_URL}/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": limit}
        return await self._request("GET", url, params=params)

    async def get_24hr_tickers(
        self, symbol: Optional[str] = None, market: str = "futures_usdm"
    ) -> List[Dict[str, Any]]:
        """Fetches 24-hour rolling window price change statistics for all or specific symbols."""
        base_url = self.FUTURES_URL if market == "futures_usdm" else self.SPOT_URL
        endpoint = "/fapi/v1/ticker/24hr" if market == "futures_usdm" else "/api/v3/ticker/24hr"
        params = {"symbol": symbol} if symbol else None
        res = await self._request("GET", f"{base_url}{endpoint}", params=params)
        return [res] if isinstance(res, dict) else res

