import re
from typing import Tuple, Optional
import yfinance as yf

# Curated list of top major cryptocurrencies
KNOWN_CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "ADA", "XRP", "DOGE", "BNB", "AVAX", "DOT", "LINK",
    "MATIC", "POL", "SHIB", "NEAR", "SUI", "PEPE", "APT", "ATOM", "XLM", "LTC",
    "BCH", "UNI", "RENDER", "FIL", "ICP", "ETC", "HBAR", "FET", "INJ", "RUNE",
    "KAS", "TIA", "TAO", "AAVE", "GRT", "THETA", "FTM", "ALGO", "SEI", "FLOW",
    "STX", "AXS", "SAND", "MANA", "EGLD", "XTZ", "EOS", "CHZ", "QNT", "CRV",
    "MKR", "SNX", "LDO", "ARB", "OP", "TRX", "TON", "WIF", "BONK", "FLOKI"
}

def canonicalize_ticker(symbol: str) -> str:
    """
    Normalizes a user-entered ticker symbol into its canonical Yahoo Finance format.
    - Strips whitespace and forces uppercase.
    - Aliases known crypto symbols (e.g., 'BTC' -> 'BTC-USD', 'ADA' -> 'ADA-USD', 'XRP' -> 'XRP-USD').
    - Leaves existing '-USD' or equity tickers intact.
    """
    if not symbol:
        return ""
    
    clean = symbol.strip().upper()
    
    # Already formatted with -USD or other currency pair
    if clean.endswith("-USD") or "-USD" in clean:
        return clean
        
    # Check against known crypto symbols
    if clean in KNOWN_CRYPTO_SYMBOLS:
        return f"{clean}-USD"
        
    # If symbol starts with crypto prefix like 'CRYPTO:BTC'
    if clean.startswith("CRYPTO:"):
        base = clean.replace("CRYPTO:", "").strip()
        return f"{base}-USD"
        
    return clean

def display_ticker(symbol: str) -> str:
    """
    Formats a canonical ticker for clean user-facing display.
    e.g. 'BTC-USD' -> 'BTC', 'ADA-USD' -> 'ADA', 'AAPL' -> 'AAPL'.
    """
    if not symbol:
        return ""
    
    clean = symbol.strip().upper()
    if clean.endswith("-USD"):
        return clean[:-4]
    return clean

def fetch_realtime_price(symbol: str, fallback_price: float = 0.0) -> Tuple[float, float, str]:
    """
    Fetches real-time price and daily change percentage for a symbol using a multi-tier fallback:
    1. fast_info (lastPrice / regularMarketPrice)
    2. info dictionary
    3. 1-day historical candles
    4. Dynamic crypto fallback: If a standard equity lookup fails or returns 0, tries '<SYMBOL>-USD'.

    Returns:
        (price: float, daily_change_pct: float, canonical_symbol: str)
    """
    canonical = canonicalize_ticker(symbol)
    
    price, change = _fetch_yf_price(canonical)
    
    # If the lookup returned 0 and it wasn't already suffixed with -USD, try dynamic crypto fallback
    if price <= 0.0 and not canonical.endswith("-USD"):
        crypto_candidate = f"{canonical}-USD"
        c_price, c_change = _fetch_yf_price(crypto_candidate)
        if c_price > 0.0:
            return c_price, c_change, crypto_candidate
            
    if price <= 0.0:
        price = fallback_price
        change = 0.0
        
    return price, change, canonical

def _fetch_yf_price(ticker_str: str) -> Tuple[float, float]:
    """Internal helper to query Yahoo Finance for price and daily change."""
    if not ticker_str:
        return 0.0, 0.0
        
    try:
        t_obj = yf.Ticker(ticker_str)
        fast = t_obj.fast_info
        
        price = 0.0
        daily_change = 0.0
        
        # 1. Try fast_info
        if fast:
            price = float(fast.get("lastPrice") or fast.get("last_price") or fast.get("regularMarketPrice") or 0.0)
            
        # 2. Try info dict fallback
        if price <= 0.0:
            try:
                info = t_obj.info or {}
                price = float(info.get("regularMarketPrice") or info.get("currentPrice") or 0.0)
                daily_change = float(info.get("regularMarketChangePercent") or 0.0)
            except Exception:
                pass
        else:
            # Try getting daily change from fast_info or info
            try:
                prev_close = float(fast.get("previousClose") or fast.get("regularMarketPreviousClose") or 0.0)
                if prev_close > 0.0 and price > 0.0:
                    daily_change = ((price - prev_close) / prev_close) * 100.0
                else:
                    info = t_obj.info or {}
                    daily_change = float(info.get("regularMarketChangePercent") or 0.0)
            except Exception:
                pass

        # 3. Try 1-day history close fallback
        if price <= 0.0:
            hist = t_obj.history(period="1d")
            if not hist.empty and "Close" in hist.columns:
                price = float(hist["Close"].iloc[-1])
                if "Open" in hist.columns and hist["Open"].iloc[-1] > 0:
                    daily_change = ((price - hist["Open"].iloc[-1]) / hist["Open"].iloc[-1]) * 100.0

        return price, daily_change
    except Exception:
        return 0.0, 0.0
