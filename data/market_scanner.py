"""Market Scanner for Binance Futures USDⓈ-M.

Monitors the full universe of USDT perpetual contracts and computes:
1. Top 10 Highest 24h Liquidity / Volume (Institutional Interest)
2. Top 10 24h Gainers (High Momentum / Expansion)
3. Top 10 24h Losers (Capitulation / Extreme Oversold Reversal Candidates)

Implements 60-second in-memory caching to strictly respect Binance REST weight limits.
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set
from data.binance_rest import BinanceRestClient

logger = logging.getLogger(__name__)


class MarketScanner:
    """Scans Binance Futures USDⓈ-M tickers and ranks assets by volume, gainers, and losers."""

    def __init__(self, rest_client: Optional[BinanceRestClient] = None, cache_ttl: float = 60.0):
        self.rest_client = rest_client or BinanceRestClient()
        self._owns_client = rest_client is None
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_time: float = 0.0
        self._all_symbols: Set[str] = set()

    async def close(self):
        """Close client if internally owned."""
        if self._owns_client and self.rest_client:
            await self.rest_client.close()

    async def fetch_rankings(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Fetches and ranks all USDT perpetual futures contracts."""
        now = time.time()
        if not force_refresh and self._cache and (now - self._cache_time < self.cache_ttl):
            return self._cache

        try:
            raw_tickers = await self.rest_client.get_24hr_tickers(market="futures_usdm")
            if not raw_tickers:
                return self._cache or {"top_volume": [], "top_gainers": [], "top_losers": []}

            parsed_items = []
            symbols_set = set()

            for t in raw_tickers:
                sym = t.get("symbol", "")
                # Only perpetual contracts: must end in USDT, no quarterlies (e.g. BTCUSDT_241227)
                if not sym.endswith("USDT") or "_" in sym:
                    continue

                symbols_set.add(sym)

                try:
                    q_vol = float(t.get("quoteVolume", 0.0))
                    chg_pct = float(t.get("priceChangePercent", 0.0))
                    last_p = float(t.get("lastPrice", 0.0))
                    hi_p = float(t.get("highPrice", 0.0))
                    lo_p = float(t.get("lowPrice", 0.0))

                    parsed_items.append({
                        "symbol": sym,
                        "price": last_p,
                        "change_pct": round(chg_pct, 2),
                        "quote_volume": q_vol,
                        "quote_volume_formatted": self._format_volume(q_vol),
                        "high": hi_p,
                        "low": lo_p,
                    })
                except (ValueError, TypeError):
                    continue

            self._all_symbols = symbols_set

            # 1. Top 10 Liquidity / Volume (Institucionais)
            top_vol = sorted(parsed_items, key=lambda x: x["quote_volume"], reverse=True)[:10]

            # 2. Top 10 Gainers (Maiores Altas 24h)
            top_gain = sorted(parsed_items, key=lambda x: x["change_pct"], reverse=True)[:10]

            # 3. Top 10 Losers (Maiores Baixas 24h — Oportunidade de Reversão de Fundo)
            top_lose = sorted(parsed_items, key=lambda x: x["change_pct"])[:10]

            now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            self._cache = {
                "top_volume": top_vol,
                "top_gainers": top_gain,
                "top_losers": top_lose,
                "updated_at": now_str,
                "total_pairs_scanned": len(parsed_items),
            }
            self._cache_time = now
            return self._cache

        except Exception as e:
            logger.error(f"Error scanning Binance Futures 24hr tickers: {e}")
            return self._cache or {
                "top_volume": [],
                "top_gainers": [],
                "top_losers": [],
                "error": str(e),
            }

    async def validate_symbol(self, symbol: str) -> Optional[str]:
        """Verifies if a symbol exists in Binance Futures USDⓈ-M.
        
        Supports auto-detection for micro-token multipliers (e.g. PEPE -> 1000PEPEUSDT).
        Returns the canonical symbol name (str) if valid, or None if invalid.
        """
        symbol = symbol.strip().upper()
        if not self._all_symbols:
            await self.fetch_rankings()

        base = symbol[:-4] if symbol.endswith("USDT") else symbol

        candidates = [
            f"{base}USDT",
            f"1000{base}USDT",
            f"10000{base}USDT",
            f"1000000{base}USDT",
        ]
        for c in candidates:
            if c in self._all_symbols:
                return c
        return None

    async def get_all_symbols(self) -> List[str]:
        """Returns sorted list of all active USDT perpetual symbols."""
        if not self._all_symbols:
            await self.fetch_rankings()
        return sorted(list(self._all_symbols))

    @staticmethod
    def _format_volume(vol: float) -> str:
        """Formats large USDT volume into human readable string."""
        if vol >= 1_000_000_000:
            return f"${vol / 1_000_000_000:.2f}B"
        elif vol >= 1_000_000:
            return f"${vol / 1_000_000:.2f}M"
        elif vol >= 1_000:
            return f"${vol / 1_000:.2f}K"
        return f"${vol:.2f}"
