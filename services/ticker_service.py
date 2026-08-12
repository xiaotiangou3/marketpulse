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

def fetch_portfolio_history(holdings: list[dict], timeframe: str = "1D") -> dict:
    """
    Calculates aggregated historical portfolio valuation across a given timeframe.
    - 1D: Lookback 1 day from current date
    - 1W: Lookback 7 days from current date
    - 1M: Lookback 30 days from current date
    - 1Y: Lookback 365 days from current date
    - ALL: Starts from the earliest ticker added by the user to current date

    Returns:
        {
            'df': pd.DataFrame with index 'Timestamp' and column 'Portfolio Value ($)',
            'start_value': float,
            'current_value': float,
            'change_value': float,
            'change_pct': float
        }
    """
    import pandas as pd
    import datetime
    
    empty_result = {
        'df': pd.DataFrame(columns=["Portfolio Value ($)"]),
        'start_value': 0.0,
        'current_value': 0.0,
        'change_value': 0.0,
        'change_pct': 0.0
    }
    
    if not holdings:
        return empty_result
        
    # Determine the earliest creation date among active holdings
    earliest_dt = None
    for h in holdings:
        c_at = h.get("created_at")
        if c_at:
            if isinstance(c_at, str):
                try:
                    c_at = pd.to_datetime(c_at)
                except Exception:
                    pass
            if hasattr(c_at, "tzinfo") and c_at.tzinfo is not None:
                c_at = c_at.replace(tzinfo=None)
            if earliest_dt is None or c_at < earliest_dt:
                earliest_dt = c_at

    tf_upper = timeframe.upper()
    now_dt = datetime.datetime.now()
    
    standard_lookbacks = {
        "1D": 1.0,
        "1W": 7.0,
        "1M": 30.0,
        "1Y": 365.0,
        "ALL": None
    }
    lookback_days = standard_lookbacks.get(tf_upper, 1.0)
    
    if earliest_dt:
        days_since_earliest = max((now_dt - earliest_dt).total_seconds() / 86400.0, 0.01)
        if lookback_days is None:
            effective_days = days_since_earliest
            effective_start = earliest_dt
        else:
            effective_days = min(lookback_days, days_since_earliest)
            effective_start = max(now_dt - datetime.timedelta(days=lookback_days), earliest_dt)
    else:
        effective_days = lookback_days if lookback_days is not None else 365.0 * 5
        effective_start = now_dt - datetime.timedelta(days=effective_days)
        
    if effective_days <= 1.5:
        period, interval = "1d", "5m"
    elif effective_days <= 7.5:
        period, interval = "5d", "15m"
    elif effective_days <= 35.0:
        period, interval = "1mo", "1h"
    elif effective_days <= 370.0:
        period, interval = "1y", "1d"
    else:
        period, interval = "max", "1d"
    
    series_dict = {}
    
    for h in holdings:
        raw_ticker = h.get("ticker", "")
        canonical = canonicalize_ticker(raw_ticker)
        shares = float(h.get("shares", 0.0))
        if shares <= 0:
            continue
            
        try:
            t_obj = yf.Ticker(canonical)
            hist = t_obj.history(period=period, interval=interval)
            
            # For 1D fallback if empty (e.g. pre-market or weekend)
            if hist.empty and period == "1d":
                hist = t_obj.history(period="5d", interval="15m")
                if not hist.empty:
                    last_date = hist.index[-1].date()
                    hist = hist[hist.index.date == last_date]
            
            if not hist.empty and "Close" in hist.columns:
                close_series = hist["Close"].dropna()
                # Normalize timezone to tz-naive for clean alignment
                if close_series.index.tz is not None:
                    close_series.index = close_series.index.tz_convert(None)
                series_dict[canonical] = close_series * shares
        except Exception:
            pass
            
    if not series_dict:
        return empty_result
        
    # Combine all series into a single DataFrame and sum
    df_combined = pd.DataFrame(series_dict)
    df_combined = df_combined.ffill().bfill()
    portfolio_total = df_combined.sum(axis=1)
    
    # Slice series starting from effective_start (capped by earliest added ticker)
    if effective_start is not None:
        sliced = portfolio_total[portfolio_total.index >= effective_start]
        if not sliced.empty and len(sliced) >= 2:
            portfolio_total = sliced
    
    if portfolio_total.empty:
        return empty_result
        
    start_val = float(portfolio_total.iloc[0])
    current_val = float(portfolio_total.iloc[-1])
    change_val = current_val - start_val
    change_pct = (change_val / start_val * 100.0) if start_val > 0 else 0.0
    
    result_df = pd.DataFrame({
        "Portfolio Value ($)": portfolio_total
    })
    result_df.index.name = "Timestamp"
    
    return {
        'df': result_df,
        'start_value': start_val,
        'current_value': current_val,
        'change_value': change_val,
        'change_pct': change_pct
    }

