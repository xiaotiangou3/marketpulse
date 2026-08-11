import threading
import json
from typing import Optional, Dict, Any
import services.alpaca_service as alpaca_service
import services.database as database

# Thread-local storage for holding trade data produced during a turn
_trade_storage = threading.local()

def set_last_trade_result(result: Dict[str, Any]):
    _trade_storage.last_result = result

def get_last_trade_result() -> Optional[Dict[str, Any]]:
    return getattr(_trade_storage, "last_result", None)

def clear_last_trade_result():
    if hasattr(_trade_storage, "last_result"):
        _trade_storage.last_result = None

def create_sandbox_tool(name: str, initial_capital: float = 100000.0, strategy_type: Optional[str] = None) -> str:
    """
    Creates a new isolated strategy sandbox with custom starting capital. (Maximum 10 sandboxes allowed).
    
    Args:
        name: Name of the sandbox (e.g. 'NVDA RSI Reversal', 'Tech Momentum').
        initial_capital: Initial virtual capital (default $100,000.0).
        strategy_type: Optional indicator or strategy category ('rsi', 'macd', 'sma_cross', etc.).
        
    Returns:
        Confirmation message with the created sandbox details.
    """
    try:
        sandboxes = database.get_sandboxes()
        if len(sandboxes) >= 10:
            return "❌ **Sandbox Limit Reached**: Maximum limit of 10 strategy sandboxes reached. Please delete an existing sandbox first."
            
        sbx_id = database.create_sandbox(
            name=name,
            initial_capital=float(initial_capital),
            strategy_type=strategy_type
        )
        return (
            f"✅ **Strategy Sandbox Created Successfully**\n"
            f"- **Sandbox Name**: {name}\n"
            f"- **Initial Capital**: ${initial_capital:,.2f}\n"
            f"- **Strategy Type**: {strategy_type or 'General'}\n"
            f"- **Sandbox ID**: `{sbx_id}`\n"
            f"- *You can now execute and track paper trades specifically within this sandbox.*"
        )
    except Exception as e:
        return f"❌ **Failed to Create Sandbox**: {e}"

def execute_paper_trade_tool(symbol: str, qty: float, side: str = "buy", sandbox_name_or_id: Optional[str] = None) -> str:
    """
    Submits a simulated paper trade order via the Alpaca Trading API into a designated strategy sandbox sub-ledger.
    
    Args:
        symbol: The stock ticker symbol (e.g. 'NVDA', 'AAPL', 'MSFT', 'TSLA').
        qty: The number of shares to buy or sell (e.g. 5.0, 10.0).
        side: The order side: 'buy' to purchase shares or 'sell' to sell/short shares (default is 'buy').
        sandbox_name_or_id: Optional name or UUID of the target strategy sandbox.
        
    Returns:
        A formatted confirmation message with order details, target sandbox, and status.
    """
    try:
        qty_float = float(qty)
        if qty_float <= 0:
            return f"Error: Order quantity must be greater than 0. Received: {qty}"
            
        clean_symbol = symbol.upper().strip()
        clean_side = side.lower().strip()
        if clean_side not in ("buy", "sell"):
            return f"Error: Order side must be 'buy' or 'sell'. Received: '{side}'"
            
        # Check existing sandboxes
        all_sandboxes = database.get_sandboxes()
        target_sandbox_id = None
        target_sandbox_name = None
        
        if not all_sandboxes:
            return (
                "⚠️ **No Active Strategy Sandboxes Found**\n\n"
                "You currently have 0 strategy sandboxes created. Would you like me to create a new sandbox for this strategy "
                f"(e.g., *'{clean_symbol} Strategy Sandbox'* with $100,000 capital) before executing your trade?"
            )
            
        if sandbox_name_or_id:
            # Match by exact ID or name
            matched = next((s for s in all_sandboxes if str(s["sandbox_id"]) == str(sandbox_name_or_id)), None)
            if not matched:
                matched = next((s for s in all_sandboxes if sandbox_name_or_id.lower() in s["name"].lower() or s["name"].lower() in sandbox_name_or_id.lower()), None)
            if matched:
                target_sandbox_id = str(matched["sandbox_id"])
                target_sandbox_name = matched["name"]
        
        # Fallback to first available sandbox if unspecified
        if not target_sandbox_id and all_sandboxes:
            target_sandbox_id = str(all_sandboxes[0]["sandbox_id"])
            target_sandbox_name = all_sandboxes[0]["name"]
            
        order_res = alpaca_service.submit_paper_order(
            symbol=clean_symbol,
            qty=qty_float,
            side=clean_side,
            sandbox_id=target_sandbox_id,
            order_type="market",
            time_in_force="gtc"
        )
        
        if target_sandbox_name:
            order_res["sandbox_name"] = target_sandbox_name
            
        # Store in thread-local for rich UI card rendering
        set_last_trade_result(order_res)
        
        status = order_res.get("status", "accepted")
        order_id = order_res.get("order_id", "N/A")
        sbx_display = f" in **{target_sandbox_name}**" if target_sandbox_name else ""
        
        return (
            f"✅ **Paper Trade Executed Successfully**\n"
            f"- **Target Sandbox**: {target_sandbox_name or 'Default'}\n"
            f"- **Order ID**: `{order_id}`\n"
            f"- **Action**: {clean_side.upper()} {qty_float:g} shares of **{clean_symbol}**{sbx_display}\n"
            f"- **Order Type**: Market (GTC)\n"
            f"- **Status**: `{status.upper()}`\n"
            f"- **Timestamp**: {order_res.get('timestamp')}\n"
            f"- *View open positions, cash balance, and leaderboard metrics in the 🧪 **Paper Trading Sandbox** tab.*"
        )
    except Exception as e:
        error_msg = f"Failed to execute paper trade for {symbol}: {e}"
        err_res = {
            "error": error_msg,
            "symbol": symbol.upper() if symbol else "UNKNOWN",
            "qty": qty,
            "side": side
        }
        set_last_trade_result(err_res)
        return f"❌ **Trade Execution Error**: {error_msg}"
