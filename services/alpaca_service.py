import datetime
from typing import Optional, List, Dict, Any
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, ClosePositionRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import config
import services.database as database

_trading_client: Optional[TradingClient] = None

def is_alpaca_configured() -> bool:
    """Checks whether Alpaca API Key and Secret Key are configured."""
    return bool(config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY and 
                config.ALPACA_API_KEY.strip() != "your_alpaca_api_key_here" and
                config.ALPACA_SECRET_KEY.strip() != "your_alpaca_secret_key_here")

def get_trading_client() -> TradingClient:
    """Initializes or returns the cached Alpaca TradingClient instance."""
    global _trading_client
    if not is_alpaca_configured():
        raise ValueError(
            "Alpaca API credentials are not configured. "
            "Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment (.env)."
        )
    if _trading_client is None:
        _trading_client = TradingClient(
            api_key=config.ALPACA_API_KEY,
            secret_key=config.ALPACA_SECRET_KEY,
            paper=config.ALPACA_PAPER
        )
    return _trading_client

def get_account_summary() -> Dict[str, Any]:
    """
    Fetches the Alpaca Paper Trading account summary.
    Returns virtual equity, available cash, buying power, and account status.
    """
    client = get_trading_client()
    account = client.get_account()
    
    equity = float(account.equity or 0.0)
    cash = float(account.cash or 0.0)
    buying_power = float(account.buying_power or 0.0)
    portfolio_value = float(account.portfolio_value or 0.0)
    last_equity = float(account.last_equity or equity)
    
    daily_pl = equity - last_equity
    daily_pl_pct = (daily_pl / last_equity * 100.0) if last_equity > 0 else 0.0
    
    return {
        "account_id": str(account.id),
        "status": str(account.status),
        "currency": str(account.currency),
        "equity": equity,
        "cash": cash,
        "buying_power": buying_power,
        "portfolio_value": portfolio_value,
        "last_equity": last_equity,
        "daily_pl": daily_pl,
        "daily_pl_pct": daily_pl_pct,
        "pattern_day_trader": getattr(account, "pattern_day_trader", False),
        "trading_blocked": getattr(account, "trading_blocked", False)
    }

def get_open_positions() -> List[Dict[str, Any]]:
    """
    Fetches all active open positions in the Alpaca Paper Trading portfolio.
    """
    client = get_trading_client()
    positions = client.get_all_positions()
    
    serialized_positions = []
    for pos in positions:
        qty = float(pos.qty or 0.0)
        avg_entry_price = float(pos.avg_entry_price or 0.0)
        current_price = float(pos.current_price or 0.0)
        market_value = float(pos.market_value or 0.0)
        unrealized_pl = float(pos.unrealized_pl or 0.0)
        unrealized_plpc = float(pos.unrealized_plpc or 0.0) * 100.0  # Convert to percentage
        change_today = float(getattr(pos, "change_today", 0.0) or 0.0) * 100.0
        
        serialized_positions.append({
            "symbol": str(pos.symbol).upper(),
            "qty": qty,
            "side": str(pos.side),
            "avg_entry_price": avg_entry_price,
            "current_price": current_price,
            "market_value": market_value,
            "cost_basis": float(pos.cost_basis or (qty * avg_entry_price)),
            "unrealized_pl": unrealized_pl,
            "unrealized_plpc": unrealized_plpc,
            "change_today": change_today
        })
        
    return serialized_positions

import uuid
import yfinance as yf

def submit_paper_order(
    symbol: str, 
    qty: float, 
    side: str, 
    sandbox_id: Optional[str] = None,
    order_type: str = "market", 
    time_in_force: str = "gtc"
) -> Dict[str, Any]:
    """
    Submits a paper trading order to Alpaca and logs execution into CockroachDB.
    
    Args:
        symbol: Ticker symbol (e.g. 'NVDA', 'AAPL')
        qty: Number of shares (float or int, > 0)
        side: Order side ('buy' or 'sell', case-insensitive)
        sandbox_id: Optional UUID of the strategy sandbox sub-ledger
        order_type: Order type ('market' supported)
        time_in_force: Time in force ('gtc' or 'day')
    """
    client = get_trading_client()
    clean_symbol = symbol.upper().strip()
    clean_side_str = side.lower().strip()
    
    if clean_side_str not in ("buy", "sell"):
        raise ValueError(f"Invalid order side: '{side}'. Must be 'buy' or 'sell'.")
    if qty <= 0:
        raise ValueError(f"Quantity must be positive. Received: {qty}")
        
    order_side = OrderSide.BUY if clean_side_str == "buy" else OrderSide.SELL
    tif = TimeInForce.GTC if time_in_force.lower() == "gtc" else TimeInForce.DAY
    
    # Custom client order id tagging sandbox
    sbx_prefix = f"sbx_{sandbox_id[:8]}" if sandbox_id else "sbx_global"
    client_order_id = f"{sbx_prefix}_{uuid.uuid4().hex[:8]}"
    
    order_data = MarketOrderRequest(
        symbol=clean_symbol,
        qty=qty,
        side=order_side,
        time_in_force=tif,
        client_order_id=client_order_id
    )
    
    order = client.submit_order(order_data=order_data)
    
    # Extract order response data
    order_id = str(order.id)
    status = str(order.status)
    filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
    
    # Estimate execution price if not filled yet (for sub-ledger valuation)
    est_price = filled_price
    if est_price is None or est_price <= 0:
        latest = database.get_latest_prices()
        if clean_symbol in latest:
            est_price = float(latest[clean_symbol]["price"])
        if est_price is None or est_price <= 0:
            try:
                h = yf.Ticker(clean_symbol).history(period="1d")
                if not h.empty:
                    est_price = float(h["Close"].iloc[-1])
            except Exception:
                est_price = 100.0
        if est_price is None or est_price <= 0:
            est_price = 100.0
            
    raw_response = {
        "id": order_id,
        "client_order_id": client_order_id,
        "symbol": clean_symbol,
        "qty": qty,
        "filled_qty": float(order.filled_qty or 0.0),
        "filled_avg_price": filled_price,
        "side": clean_side_str,
        "type": str(order.type),
        "time_in_force": str(order.time_in_force),
        "status": status,
        "sandbox_id": sandbox_id,
        "created_at": order.created_at.isoformat() if hasattr(order.created_at, "isoformat") else str(order.created_at)
    }
    
    # Update sandbox sub-ledger in CockroachDB if sandbox_id provided
    sandbox_name = None
    if sandbox_id:
        try:
            database.update_sandbox_position_and_cash(
                sandbox_id=sandbox_id,
                symbol=clean_symbol,
                qty=qty,
                execution_price=est_price,
                side=clean_side_str
            )
            sbx = database.get_sandbox_by_id(sandbox_id)
            if sbx:
                sandbox_name = sbx["name"]
        except Exception as e:
            print(f"Warning: Failed to update sandbox sub-ledger in CockroachDB: {e}")
            
    # Persist in CockroachDB audit log
    trade_id = None
    try:
        trade_id = database.log_paper_trade(
            order_id=order_id,
            symbol=clean_symbol,
            side=clean_side_str,
            qty=qty,
            status=status,
            execution_price=filled_price or est_price,
            order_type=order_type,
            time_in_force=time_in_force,
            raw_response=raw_response,
            sandbox_id=sandbox_id
        )
    except Exception as e:
        print(f"Warning: Failed to log paper trade in CockroachDB: {e}")
        
    result = {
        "success": True,
        "trade_id": trade_id,
        "order_id": order_id,
        "client_order_id": client_order_id,
        "symbol": clean_symbol,
        "qty": qty,
        "side": clean_side_str.upper(),
        "status": status,
        "execution_price": filled_price or est_price,
        "order_type": order_type.upper(),
        "time_in_force": time_in_force.upper(),
        "sandbox_id": sandbox_id,
        "sandbox_name": sandbox_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "raw": raw_response
    }
    return result

def close_position(symbol: str) -> Dict[str, Any]:
    """Closes an open position for the specified symbol globally."""
    client = get_trading_client()
    clean_symbol = symbol.upper().strip()
    order = client.close_position(symbol_or_asset_id=clean_symbol)
    
    order_id = str(order.id)
    status = str(order.status)
    qty = float(order.qty or 0.0)
    side = str(order.side).lower()
    
    raw_response = {
        "id": order_id,
        "symbol": clean_symbol,
        "qty": qty,
        "side": side,
        "status": status
    }
    
    try:
        database.log_paper_trade(
            order_id=order_id,
            symbol=clean_symbol,
            side=side,
            qty=qty,
            status=status,
            order_type="market",
            time_in_force="day",
            raw_response=raw_response
        )
    except Exception as e:
        print(f"Warning: Failed to log close position trade in CockroachDB: {e}")
        
    return {
        "success": True,
        "order_id": order_id,
        "symbol": clean_symbol,
        "qty": qty,
        "side": side.upper(),
        "status": status
    }

def close_sandbox_position(sandbox_id: str, symbol: str) -> Dict[str, Any]:
    """Closes an open position specifically within a designated sandbox sub-ledger."""
    clean_symbol = symbol.upper().strip()
    positions = database.get_sandbox_positions(sandbox_id)
    pos = next((p for p in positions if p["symbol"] == clean_symbol), None)
    
    if not pos:
        raise ValueError(f"No active position for {clean_symbol} found in sandbox {sandbox_id}.")
        
    pos_qty = float(pos["qty"])
    if pos_qty <= 0:
        raise ValueError(f"Position quantity for {clean_symbol} is zero.")
        
    return submit_paper_order(
        symbol=clean_symbol,
        qty=pos_qty,
        side="sell",
        sandbox_id=sandbox_id,
        order_type="market",
        time_in_force="gtc"
    )

