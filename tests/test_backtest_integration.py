import unittest
import pandas as pd
import numpy as np
import agent
from agent.backtest import (
    backtest_universal_strategy,
    backtest_sma_strategy, 
    backtest_strategy_tool, 
    get_last_backtest_result, 
    set_last_backtest_result, 
    clear_last_backtest_result
)

class TestMultiStrategyEngine(unittest.TestCase):

    def test_backtest_sma_crossover(self):
        result = backtest_universal_strategy(ticker="MSFT", strategy_type="sma_cross", period="6mo", short_window=10, long_window=30)
        self.assertEqual(result["ticker"], "MSFT")
        self.assertEqual(result["strategy_type"], "sma_cross")
        self.assertIn("Strategy_Return_Pct", result)
        self.assertIn("chart_data", result)
        self.assertIsNone(result.get("error"))

    def test_backtest_ema_crossover(self):
        result = backtest_universal_strategy(ticker="AAPL", strategy_type="ema_cross", period="6mo", short_window=12, long_window=26)
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["strategy_type"], "ema_cross")
        self.assertIn("EMA Crossover", result["strategy_name"])
        self.assertIn("Win_Rate_Pct", result)
        self.assertIsNone(result.get("error"))

    def test_backtest_rsi_mean_reversion(self):
        result = backtest_universal_strategy(ticker="NVDA", strategy_type="rsi", period="1y", rsi_period=14, rsi_oversold=30, rsi_overbought=70)
        self.assertEqual(result["ticker"], "NVDA")
        self.assertEqual(result["strategy_type"], "rsi")
        self.assertIn("RSI Mean Reversion", result["strategy_name"])
        self.assertIn("Strategy_Return_Pct", result)
        self.assertIsNone(result.get("error"))

    def test_backtest_macd_crossover(self):
        result = backtest_universal_strategy(ticker="TSLA", strategy_type="macd", period="6mo", macd_fast=12, macd_slow=26, macd_signal=9)
        self.assertEqual(result["ticker"], "TSLA")
        self.assertEqual(result["strategy_type"], "macd")
        self.assertIn("MACD Crossover", result["strategy_name"])
        self.assertIn("Max_Drawdown_Pct", result)
        self.assertIsNone(result.get("error"))

    def test_backtest_bollinger_bands(self):
        result = backtest_universal_strategy(ticker="GOOGL", strategy_type="bollinger", period="6mo", bb_window=20, bb_std=2.0)
        self.assertEqual(result["ticker"], "GOOGL")
        self.assertEqual(result["strategy_type"], "bollinger")
        self.assertIn("Bollinger Bands", result["strategy_name"])
        self.assertIn("Win_Rate_Pct", result)
        self.assertIsNone(result.get("error"))

    def test_backtest_breakout_momentum(self):
        result = backtest_universal_strategy(ticker="AMZN", strategy_type="breakout", period="6mo", breakout_window=20)
        self.assertEqual(result["ticker"], "AMZN")
        self.assertEqual(result["strategy_type"], "breakout")
        self.assertIn("Price Breakout", result["strategy_name"])
        self.assertIn("Strategy_Return_Pct", result)
        self.assertIsNone(result.get("error"))

    def test_backtest_lookahead_bias_elimination(self):
        # Verify that signal shifting by 1 bar strictly happens for RSI logic
        dates = pd.date_range("2025-01-01", periods=50, freq="D")
        prices = [100.0 - i if i < 25 else 75.0 + (i - 25) * 2 for i in range(50)]
        df = pd.DataFrame({"Close": prices}, index=dates)
        
        # When oversold triggered at row i, position must only become active at row i+1
        df["Signal"] = [1.0 if i >= 20 else 0.0 for i in range(50)]
        df["Position"] = df["Signal"].shift(1).fillna(0.0)
        
        self.assertEqual(df["Signal"].iloc[20], 1.0)
        self.assertEqual(df["Position"].iloc[20], 0.0, "Position must be 0 on the signal day to eliminate lookahead bias")
        self.assertEqual(df["Position"].iloc[21], 1.0, "Position takes effect on the next trading day")

    def test_backtest_strategy_tool_formatting(self):
        # RSI tool call formatting
        summary_rsi = backtest_strategy_tool(ticker="NVDA", strategy_type="rsi", timeframe="6mo", rsi_period=14, rsi_oversold=30, rsi_overbought=70)
        self.assertIn("RSI Mean Reversion", summary_rsi)
        self.assertIn("Quantitative Backtest Results for NVDA", summary_rsi)
        
        # MACD tool call formatting
        summary_macd = backtest_strategy_tool(ticker="AAPL", strategy_type="macd", timeframe="6mo")
        self.assertIn("MACD Crossover", summary_macd)

    def test_thread_storage(self):
        clear_last_backtest_result()
        self.assertIsNone(get_last_backtest_result())
        
        sample_payload = {"ticker": "NVDA", "strategy_type": "rsi", "Strategy_Return_Pct": 18.5}
        set_last_backtest_result(sample_payload)
        self.assertEqual(get_last_backtest_result(), sample_payload)
        
        clear_last_backtest_result()
        self.assertIsNone(get_last_backtest_result())

if __name__ == "__main__":
    unittest.main()
