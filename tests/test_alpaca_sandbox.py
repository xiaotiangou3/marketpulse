import unittest
from unittest.mock import patch, MagicMock
import datetime
import uuid
import services.database as database
import services.alpaca_service as alpaca_service
import services
import agent
from agent.trade_tools import (
    execute_paper_trade_tool,
    get_last_trade_result,
    set_last_trade_result,
    clear_last_trade_result
)

class TestAlpacaPaperTradingSandbox(unittest.TestCase):

    def setUp(self):
        clear_last_trade_result()

    def tearDown(self):
        clear_last_trade_result()

    def test_trade_thread_storage(self):
        self.assertIsNone(get_last_trade_result())
        sample = {"order_id": "test-123", "symbol": "NVDA", "qty": 10.0, "side": "BUY"}
        set_last_trade_result(sample)
        self.assertEqual(get_last_trade_result(), sample)
        clear_last_trade_result()
        self.assertIsNone(get_last_trade_result())

    def test_database_paper_trade_logging(self):
        unique_order_id = f"test-order-{uuid.uuid4().hex[:8]}"
        trade_id = database.log_paper_trade(
            order_id=unique_order_id,
            symbol="NVDA",
            side="buy",
            qty=15.0,
            status="accepted",
            execution_price=125.50,
            order_type="market",
            time_in_force="gtc",
            raw_response={"client_order_id": "test_cli_id"}
        )
        self.assertIsNotNone(trade_id)
        
        # Verify in retrieval
        logs = database.get_paper_trade_logs(limit=10)
        self.assertTrue(len(logs) > 0)
        matched = next((l for l in logs if l["order_id"] == unique_order_id), None)
        self.assertIsNotNone(matched)
        self.assertEqual(matched["symbol"], "NVDA")
        self.assertEqual(matched["side"], "BUY")
        self.assertEqual(float(matched["qty"]), 15.0)
        self.assertEqual(float(matched["execution_price"]), 125.50)
        self.assertEqual(matched["status"], "accepted")

    def test_submit_paper_order_validation(self):
        # Invalid side
        with self.assertRaises(ValueError):
            alpaca_service.submit_paper_order(symbol="AAPL", qty=10, side="hold")
            
        # Invalid quantity <= 0
        with self.assertRaises(ValueError):
            alpaca_service.submit_paper_order(symbol="AAPL", qty=-5, side="buy")

    @patch("services.alpaca_service.get_trading_client")
    @patch("services.alpaca_service.is_alpaca_configured", return_value=True)
    def test_submit_paper_order_mock(self, mock_configured, mock_get_client):
        mock_client = MagicMock()
        mock_order = MagicMock()
        mock_order.id = uuid.uuid4()
        mock_order.client_order_id = "test-client-id"
        mock_order.symbol = "NVDA"
        mock_order.qty = 10.0
        mock_order.filled_qty = 10.0
        mock_order.filled_avg_price = 120.50
        mock_order.side = "buy"
        mock_order.type = "market"
        mock_order.time_in_force = "gtc"
        mock_order.status = "filled"
        mock_order.created_at = datetime.datetime.now(datetime.timezone.utc)
        
        mock_client.submit_order.return_value = mock_order
        mock_get_client.return_value = mock_client
        
        result = alpaca_service.submit_paper_order(symbol="NVDA", qty=10.0, side="buy")
        self.assertTrue(result["success"])
        self.assertEqual(result["symbol"], "NVDA")
        self.assertEqual(result["side"], "BUY")
        self.assertEqual(result["qty"], 10.0)
        self.assertEqual(result["status"], "filled")
        self.assertEqual(result["execution_price"], 120.50)

    @patch("services.alpaca_service.get_trading_client")
    @patch("services.alpaca_service.is_alpaca_configured", return_value=True)
    def test_get_account_summary_mock(self, mock_configured, mock_get_client):
        mock_client = MagicMock()
        mock_account = MagicMock()
        mock_account.id = uuid.uuid4()
        mock_account.status = "ACTIVE"
        mock_account.currency = "USD"
        mock_account.equity = "105000.50"
        mock_account.cash = "95000.00"
        mock_account.buying_power = "200000.00"
        mock_account.portfolio_value = "105000.50"
        mock_account.last_equity = "100000.00"
        mock_account.pattern_day_trader = False
        mock_account.trading_blocked = False
        
        mock_client.get_account.return_value = mock_account
        mock_get_client.return_value = mock_client
        
        summary = alpaca_service.get_account_summary()
        self.assertEqual(summary["status"], "ACTIVE")
        self.assertEqual(summary["equity"], 105000.50)
        self.assertEqual(summary["cash"], 95000.00)
        self.assertEqual(summary["buying_power"], 200000.00)
        self.assertEqual(summary["daily_pl"], 5000.50)

    @patch("services.alpaca_service.get_trading_client")
    @patch("services.alpaca_service.is_alpaca_configured", return_value=True)
    def test_get_open_positions_mock(self, mock_configured, mock_get_client):
        mock_client = MagicMock()
        mock_pos = MagicMock()
        mock_pos.symbol = "NVDA"
        mock_pos.qty = "10"
        mock_pos.side = "long"
        mock_pos.avg_entry_price = "110.00"
        mock_pos.current_price = "125.00"
        mock_pos.market_value = "1250.00"
        mock_pos.cost_basis = "1100.00"
        mock_pos.unrealized_pl = "150.00"
        mock_pos.unrealized_plpc = "0.1363"
        mock_pos.change_today = "0.025"
        
        mock_client.get_all_positions.return_value = [mock_pos]
        mock_get_client.return_value = mock_client
        
        positions = alpaca_service.get_open_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["symbol"], "NVDA")
        self.assertEqual(positions[0]["qty"], 10.0)
        self.assertEqual(positions[0]["unrealized_pl"], 150.00)
        self.assertAlmostEqual(positions[0]["unrealized_plpc"], 13.63, places=2)

    @patch("services.database.get_sandboxes")
    @patch("services.alpaca_service.submit_paper_order")
    def test_execute_paper_trade_tool(self, mock_submit, mock_get_sbx):
        # 1. Zero sandboxes case
        mock_get_sbx.return_value = []
        zero_msg = execute_paper_trade_tool(symbol="AAPL", qty=5.0, side="buy")
        self.assertIn("No Active Strategy Sandboxes Found", zero_msg)
        
        # 2. Existing sandbox case
        mock_get_sbx.return_value = [{"sandbox_id": "test-sbx-uuid", "name": "General Sandbox"}]
        mock_submit.return_value = {
            "success": True,
            "order_id": "alpaca-ord-999",
            "symbol": "AAPL",
            "qty": 5.0,
            "side": "BUY",
            "status": "accepted",
            "execution_price": None,
            "sandbox_name": "General Sandbox",
            "timestamp": "2026-08-11 22:00:00 UTC"
        }
        
        msg = execute_paper_trade_tool(symbol="AAPL", qty=5.0, side="buy")
        self.assertIn("Paper Trade Executed Successfully", msg)
        self.assertIn("BUY 5 shares of **AAPL**", msg)
        self.assertIn("alpaca-ord-999", msg)
        
        cached = get_last_trade_result()
        self.assertIsNotNone(cached)
        self.assertEqual(cached["symbol"], "AAPL")
        self.assertEqual(cached["qty"], 5.0)


    def test_router_paper_trade_intent(self):
        # Direct trade prompt
        output = agent.route_user_intent(
            user_prompt="Buy 10 shares of NVDA",
            has_uploaded_file=False
        )
        action_types = [a.action_type for a in output.actions]
        self.assertIn("paper_trade", action_types)
        trade_action = next(a for a in output.actions if a.action_type == "paper_trade")
        self.assertEqual(trade_action.ticker, "NVDA")
        self.assertEqual(trade_action.qty, 10.0)
        self.assertEqual(trade_action.side, "buy")

if __name__ == "__main__":
    unittest.main()
