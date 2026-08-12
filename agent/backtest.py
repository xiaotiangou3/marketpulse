import threading
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Any

# Thread-local storage for holding backtest data produced during a turn
_backtest_storage = threading.local()

def set_last_backtest_result(result: Dict[str, Any]):
    _backtest_storage.last_result = result

def get_last_backtest_result() -> Optional[Dict[str, Any]]:
    return getattr(_backtest_storage, "last_result", None)

def clear_last_backtest_result():
    if hasattr(_backtest_storage, "last_result"):
        _backtest_storage.last_result = None

def compute_rsi_series(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes Relative Strength Index (RSI) using Wilder's smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)

def backtest_universal_strategy(
    ticker: str,
    strategy_type: str = "sma_cross",
    period: str = "1y",
    short_window: int = 20,
    long_window: int = 50,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
    breakout_window: int = 20
) -> Dict[str, Any]:
    """
    Universal Quantitative Backtesting Engine.
    Executes strategy simulations on daily historical closing prices with strictly lagged signals.
    """
    ticker = ticker.upper().strip()
    strat_type = strategy_type.lower().strip()
    
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period)
        
        if df.empty:
            res = {
                "ticker": ticker,
                "period": period,
                "strategy_type": strat_type,
                "error": f"No historical market data found for ticker '{ticker}'."
            }
            set_last_backtest_result(res)
            return res
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if "Close" not in df.columns or len(df) < 30:
            res = {
                "ticker": ticker,
                "period": period,
                "strategy_type": strat_type,
                "error": f"Insufficient historical data points ({len(df)} rows) to compute indicators."
            }
            set_last_backtest_result(res)
            return res

        df["Signal"] = 0.0
        strategy_name = "Strategy"
        condition_summary = ""

        # --- 1. SMA CROSSOVER ---
        if strat_type in ["sma_cross", "sma", "moving_average"]:
            strategy_name = f"SMA Crossover ({short_window}/{long_window})"
            condition_summary = f"Buy when {short_window}-day SMA > {long_window}-day SMA, Cash otherwise"
            df["SMA_Fast"] = df["Close"].rolling(window=short_window).mean()
            df["SMA_Slow"] = df["Close"].rolling(window=long_window).mean()
            df.loc[df["SMA_Fast"] > df["SMA_Slow"], "Signal"] = 1.0

        # --- 2. EMA CROSSOVER ---
        elif strat_type in ["ema_cross", "ema", "exponential_moving_average"]:
            strategy_name = f"EMA Crossover ({short_window}/{long_window})"
            condition_summary = f"Buy when {short_window}-day EMA > {long_window}-day EMA, Cash otherwise"
            df["EMA_Fast"] = df["Close"].ewm(span=short_window, adjust=False).mean()
            df["EMA_Slow"] = df["Close"].ewm(span=long_window, adjust=False).mean()
            df.loc[df["EMA_Fast"] > df["EMA_Slow"], "Signal"] = 1.0

        # --- 3. RSI MEAN REVERSION ---
        elif strat_type in ["rsi", "rsi_mean_reversion", "relative_strength"]:
            strategy_name = f"RSI Mean Reversion ({rsi_period}p, {rsi_oversold:.0f}/{rsi_overbought:.0f})"
            condition_summary = f"Buy when RSI({rsi_period}) < {rsi_oversold}, Exit when RSI({rsi_period}) > {rsi_overbought}"
            df["RSI"] = compute_rsi_series(df["Close"], period=rsi_period)
            
            # State-tracking for entry/exit
            curr_pos = 0.0
            signals = []
            for rsi_val in df["RSI"]:
                if rsi_val < rsi_oversold:
                    curr_pos = 1.0
                elif rsi_val > rsi_overbought:
                    curr_pos = 0.0
                signals.append(curr_pos)
            df["Signal"] = signals

        # --- 4. MACD CROSSOVER ---
        elif strat_type in ["macd", "macd_cross"]:
            strategy_name = f"MACD Crossover ({macd_fast}/{macd_slow}/{macd_signal})"
            condition_summary = f"Buy when MACD Line > Signal Line ({macd_signal} EMA), Cash otherwise"
            ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
            ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
            df["MACD_Line"] = ema_fast - ema_slow
            df["MACD_Signal"] = df["MACD_Line"].ewm(span=macd_signal, adjust=False).mean()
            df.loc[df["MACD_Line"] > df["MACD_Signal"], "Signal"] = 1.0

        # --- 5. BOLLINGER BANDS ---
        elif strat_type in ["bollinger", "bollinger_bands", "bb"]:
            strategy_name = f"Bollinger Bands Dip Buying ({bb_window}p, {bb_std}σ)"
            condition_summary = f"Buy when Close < Lower Band ({bb_std}σ), Exit when Close > Upper Band"
            df["BB_Mid"] = df["Close"].rolling(window=bb_window).mean()
            df["BB_Std"] = df["Close"].rolling(window=bb_window).std()
            df["BB_Upper"] = df["BB_Mid"] + (bb_std * df["BB_Std"])
            df["BB_Lower"] = df["BB_Mid"] - (bb_std * df["BB_Std"])
            
            curr_pos = 0.0
            signals = []
            for close_p, upper_p, lower_p in zip(df["Close"], df["BB_Upper"], df["BB_Lower"]):
                if close_p < lower_p:
                    curr_pos = 1.0
                elif close_p > upper_p:
                    curr_pos = 0.0
                signals.append(curr_pos)
            df["Signal"] = signals

        # --- 6. PRICE BREAKOUT / DONCHIAN CHANNEL ---
        elif strat_type in ["breakout", "donchian", "channel_breakout", "momentum"]:
            strategy_name = f"Price Breakout ({breakout_window}-day Channel)"
            condition_summary = f"Buy on {breakout_window}-day High Breakout, Exit on {breakout_window}-day Low Breakdown"
            df["Highest_High"] = df["Close"].shift(1).rolling(window=breakout_window).max()
            df["Lowest_Low"] = df["Close"].shift(1).rolling(window=breakout_window).min()
            
            curr_pos = 0.0
            signals = []
            for close_p, h_high, l_low in zip(df["Close"], df["Highest_High"], df["Lowest_Low"]):
                if pd.notna(h_high) and close_p > h_high:
                    curr_pos = 1.0
                elif pd.notna(l_low) and close_p < l_low:
                    curr_pos = 0.0
                signals.append(curr_pos)
            df["Signal"] = signals

        else:
            return {
                "ticker": ticker,
                "period": period,
                "strategy_type": strat_type,
                "error": f"Unknown strategy type '{strat_type}'. Supported types: sma_cross, ema_cross, rsi, macd, bollinger, breakout."
            }

        # Shift position by 1 day to strictly eliminate lookahead bias
        df["Position"] = df["Signal"].shift(1).fillna(0.0)
        
        # Calculate daily returns
        df["Market_Return"] = df["Close"].pct_change().fillna(0.0)
        df["Strategy_Return"] = df["Position"] * df["Market_Return"]
        
        # Cumulative returns
        df["Cumulative_Market"] = (1.0 + df["Market_Return"]).cumprod()
        df["Cumulative_Strategy"] = (1.0 + df["Strategy_Return"]).cumprod()
        
        strategy_return_pct = float((df["Cumulative_Strategy"].iloc[-1] - 1.0) * 100.0)
        buy_hold_return_pct = float((df["Cumulative_Market"].iloc[-1] - 1.0) * 100.0)
        
        # Max Drawdown
        cum_strat = df["Cumulative_Strategy"]
        running_max = cum_strat.cummax()
        drawdown = (cum_strat - running_max) / running_max
        max_drawdown_pct = float(abs(drawdown.min() * 100.0))
        
        # Win Rate (% of invested days with positive market returns)
        active_days = df[df["Position"] == 1.0]
        if len(active_days) > 0:
            winning_days = len(active_days[active_days["Market_Return"] > 0])
            win_rate_pct = float((winning_days / len(active_days)) * 100.0)
        else:
            win_rate_pct = 0.0
            
        outperformed = strategy_return_pct > buy_hold_return_pct
        
        # Chart Data
        chart_records = []
        for dt, row in df.iterrows():
            date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
            chart_records.append({
                "Date": date_str,
                "Strategy Return (%)": round(float((row["Cumulative_Strategy"] - 1.0) * 100.0), 2),
                "Buy & Hold (%)": round(float((row["Cumulative_Market"] - 1.0) * 100.0), 2)
            })
            
        result = {
            "ticker": ticker,
            "period": period,
            "strategy_type": strat_type,
            "strategy_name": strategy_name,
            "condition_summary": condition_summary,
            "short_window": short_window,
            "long_window": long_window,
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": macd_signal,
            "bb_window": bb_window,
            "bb_std": bb_std,
            "breakout_window": breakout_window,
            "Strategy_Return_Pct": round(strategy_return_pct, 2),
            "Buy_Hold_Return_Pct": round(buy_hold_return_pct, 2),
            "Max_Drawdown_Pct": round(max_drawdown_pct, 2),
            "Win_Rate_Pct": round(win_rate_pct, 2),
            "outperformed": outperformed,
            "total_bars": len(df),
            "chart_data": chart_records,
            "error": None
        }
        
        set_last_backtest_result(result)
        return result
        
    except Exception as e:
        err_res = {
            "ticker": ticker,
            "period": period,
            "strategy_type": strat_type,
            "error": f"Backtest execution failed: {str(e)}"
        }
        set_last_backtest_result(err_res)
        return err_res

def backtest_sma_strategy(ticker: str, period: str = "1y", short_window: int = 20, long_window: int = 50) -> Dict[str, Any]:
    """Backward-compatible helper for SMA crossover backtests."""
    return backtest_universal_strategy(
        ticker=ticker,
        strategy_type="sma_cross",
        period=period,
        short_window=short_window,
        long_window=long_window
    )

def backtest_strategy_tool(
    ticker: str,
    strategy_type: str = "sma_cross",
    timeframe: str = "1y",
    short_window: int = 20,
    long_window: int = 50,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_window: int = 20,
    bb_std: float = 2.0,
    breakout_window: int = 20
) -> str:
    """
    Backtests a quantitative trading strategy on historical price data with zero lookahead bias.
    
    Supported strategy_type values:
    - 'sma_cross': Simple Moving Average crossover (uses short_window, long_window).
    - 'ema_cross': Exponential Moving Average crossover (uses short_window, long_window).
    - 'rsi': RSI mean reversion / oversold buying (uses rsi_period, rsi_oversold, rsi_overbought).
    - 'macd': MACD line crossing above signal line (uses macd_fast, macd_slow, macd_signal).
    - 'bollinger': Bollinger Bands lower band dip buying / upper band exit (uses bb_window, bb_std).
    - 'breakout': Price high/low channel breakout momentum (uses breakout_window).
    
    Args:
        ticker: Stock ticker symbol (e.g. 'NVDA', 'AAPL', 'MSFT', 'TSLA').
        strategy_type: One of 'sma_cross', 'ema_cross', 'rsi', 'macd', 'bollinger', 'breakout'.
        timeframe: Historical lookback period ('6mo', '1y', '2y', '5y'). Default is '1y'.
        short_window: Lookback period for fast moving average (default 20).
        long_window: Lookback period for slow moving average (default 50).
        rsi_period: Lookback period for RSI (default 14).
        rsi_oversold: RSI buy threshold (default 30.0).
        rsi_overbought: RSI exit threshold (default 70.0).
        macd_fast: MACD fast EMA span (default 12).
        macd_slow: MACD slow EMA span (default 26).
        macd_signal: MACD signal line EMA span (default 9).
        bb_window: Bollinger Bands moving average period (default 20).
        bb_std: Bollinger Bands standard deviation multiplier (default 2.0).
        breakout_window: Lookback period for high/low breakout channel (default 20).
        
    Returns:
        A formatted markdown summary of the strategy backtest metrics.
    """
    res = backtest_universal_strategy(
        ticker=ticker,
        strategy_type=strategy_type,
        period=timeframe,
        short_window=short_window,
        long_window=long_window,
        rsi_period=rsi_period,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
        macd_fast=macd_fast,
        macd_slow=macd_slow,
        macd_signal=macd_signal,
        bb_window=bb_window,
        bb_std=bb_std,
        breakout_window=breakout_window
    )
    
    if res.get("error"):
        return f"Backtest Error for {ticker} ({strategy_type}): {res['error']}"
        
    status_text = "Outperformed Benchmark" if res["outperformed"] else "Underperformed Benchmark"
    
    summary = (
        f"### Quantitative Backtest Results for {res['ticker']}:\n"
        f"- **Strategy**: {res['strategy_name']}\n"
        f"- **Rules**: {res['condition_summary']}\n"
        f"- **Timeframe**: {res['period']}\n"
        f"- **Strategy Total Return**: {res['Strategy_Return_Pct']:+.2f}%\n"
        f"- **Buy & Hold Benchmark Return**: {res['Buy_Hold_Return_Pct']:+.2f}%\n"
        f"- **Win Rate**: {res['Win_Rate_Pct']:.2f}%\n"
        f"- **Max Drawdown**: {res['Max_Drawdown_Pct']:.2f}%\n"
        f"- **Outcome**: {status_text} (Positions lagged 1 day to strictly eliminate lookahead bias)."
    )
    return summary
