"""Dynamic symbol discovery and universe ranking for Binance Spot and Futures.

Filters pairs based on:
- Active trading status
- Base/Quote asset criteria (e.g., USDT, USDC)
- Minimum 24h rolling volume
- Bid/ask spread and order book depth
- Contract type (Perpetual vs Quarterly)
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from data.binance_rest import BinanceRestClient

logger = logging.getLogger(__name__)


class SymbolMetadata(BaseModel):
    symbol: str
    base_asset: str
    quote_asset: str
    market: str  # "futures_usdm" or "spot"
    status: str
    price_precision: int
    quantity_precision: int
    volume_24h_usd: float = 0.0
    spread_pct: float = 0.0
    has_open_interest: bool = False
    has_funding: bool = False


class SymbolUniverse:
    """Manages dynamic symbol discovery, liquidity filtering, and rankings."""

    def __init__(
        self,
        rest_client: BinanceRestClient,
        quote_assets: Optional[List[str]] = None,
        min_volume_24h_usd: float = 20_000_000.0,
        max_spread_pct: float = 0.05,
    ):
        self.client = rest_client
        self.quote_assets = quote_assets or ["USDT", "USDC"]
        self.min_volume_24h_usd = min_volume_24h_usd
        self.max_spread_pct = max_spread_pct

    async def discover_futures_symbols(
        self,
        top_n: Optional[int] = None,
        custom_symbols: Optional[List[str]] = None,
    ) -> List[SymbolMetadata]:
        """Discovers active USDⓈ-M Perpetual Futures pairs passing liquidity filters."""
        exchange_info = await self.client.get_exchange_info(market="futures_usdm")
        tickers_24h = await self.client.get_24hr_ticker(market="futures_usdm")

        ticker_map = {t["symbol"]: t for t in tickers_24h if "symbol" in t}

        eligible_symbols: List[SymbolMetadata] = []

        for item in exchange_info.get("symbols", []):
            symbol = item["symbol"]
            status = item.get("status", "")
            contract_type = item.get("contractType", "")

            # Filter perpetual and trading
            if status != "TRADING" or contract_type != "PERPETUAL":
                continue

            quote_asset = item.get("quoteAsset", "")
            if quote_asset not in self.quote_assets:
                continue

            if custom_symbols and symbol not in custom_symbols:
                continue

            ticker = ticker_map.get(symbol, {})
            volume_24h = float(ticker.get("quoteVolume", 0.0))
            last_price = float(ticker.get("lastPrice", 0.0))

            # Spread estimation from best bid/ask in ticker or ticker snapshot
            bid_price = float(ticker.get("bidPrice", 0.0))
            ask_price = float(ticker.get("askPrice", 0.0))
            spread_pct = 0.0
            if bid_price > 0 and ask_price >= bid_price:
                spread_pct = ((ask_price - bid_price) / bid_price) * 100.0

            # Filter by volume and spread if not custom
            if not custom_symbols:
                if volume_24h < self.min_volume_24h_usd:
                    continue
                if spread_pct > self.max_spread_pct:
                    continue

            meta = SymbolMetadata(
                symbol=symbol,
                base_asset=item.get("baseAsset", ""),
                quote_asset=quote_asset,
                market="futures_usdm",
                status=status,
                price_precision=int(item.get("pricePrecision", 2)),
                quantity_precision=int(item.get("quantityPrecision", 2)),
                volume_24h_usd=volume_24h,
                spread_pct=spread_pct,
                has_open_interest=True,
                has_funding=True,
            )
            eligible_symbols.append(meta)

        # Sort descending by 24h volume
        eligible_symbols.sort(key=lambda s: s.volume_24h_usd, reverse=True)

        if top_n is not None and top_n > 0:
            return eligible_symbols[:top_n]

        return eligible_symbols

    async def discover_spot_symbols(
        self,
        top_n: Optional[int] = None,
        custom_symbols: Optional[List[str]] = None,
    ) -> List[SymbolMetadata]:
        """Discovers active Spot pairs passing liquidity filters."""
        exchange_info = await self.client.get_exchange_info(market="spot")
        tickers_24h = await self.client.get_24hr_ticker(market="spot")

        ticker_map = {t["symbol"]: t for t in tickers_24h if "symbol" in t}
        eligible_symbols: List[SymbolMetadata] = []

        for item in exchange_info.get("symbols", []):
            symbol = item["symbol"]
            status = item.get("status", "")

            if status != "TRADING":
                continue

            quote_asset = item.get("quoteAsset", "")
            if quote_asset not in self.quote_assets:
                continue

            if custom_symbols and symbol not in custom_symbols:
                continue

            ticker = ticker_map.get(symbol, {})
            volume_24h = float(ticker.get("quoteVolume", 0.0))
            bid_price = float(ticker.get("bidPrice", 0.0))
            ask_price = float(ticker.get("askPrice", 0.0))
            spread_pct = 0.0
            if bid_price > 0 and ask_price >= bid_price:
                spread_pct = ((ask_price - bid_price) / bid_price) * 100.0

            if not custom_symbols:
                if volume_24h < self.min_volume_24h_usd:
                    continue
                if spread_pct > self.max_spread_pct:
                    continue

            meta = SymbolMetadata(
                symbol=symbol,
                base_asset=item.get("baseAsset", ""),
                quote_asset=quote_asset,
                market="spot",
                status=status,
                price_precision=int(item.get("quotePrecision", 2)),
                quantity_precision=int(item.get("baseAssetPrecision", 2)),
                volume_24h_usd=volume_24h,
                spread_pct=spread_pct,
                has_open_interest=False,
                has_funding=False,
            )
            eligible_symbols.append(meta)

        eligible_symbols.sort(key=lambda s: s.volume_24h_usd, reverse=True)

        if top_n is not None and top_n > 0:
            return eligible_symbols[:top_n]

        return eligible_symbols

