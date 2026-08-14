import unittest
from unittest.mock import patch, MagicMock
import uuid
import services.database as database
import services.alpaca_service as alpaca_service
import services
import agent
from agent.trade_tools import (
    create_sandbox_tool,
    execute_paper_trade_tool,
    get_last_trade_result,
    clear_last_trade_result
)

class TestMultiStrategyPaperTrading(unittest.TestCase):

    def setUp(self):
        clear_last_trade_result()
        # Clean up demo sandboxes before each test
        conn = database.get_db_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sandboxes WHERE user_id = 'test_multi_user';")
            conn.commit()
        database.release_db_connection(conn)

    def tearDown(self):
        clear_last_trade_result()
        conn = database.get_db_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sandboxes WHERE user_id = 'test_multi_user';")
            conn.commit()
        database.release_db_connection(conn)

    def test_create_and_get_sandboxes(self):
        # Create 2 sandboxes
        sbx1_id = database.create_sandbox(
            name="RSI Mean Reversion",
            description="Trades NVDA RSI",
            initial_capital=50000.0,
            strategy_type="rsi",
            user_id="test_multi_user"
        )
        self.assertIsNotNone(sbx1_id)
        
        sbx2_id = database.create_sandbox(
            name="Tech Momentum",
            initial_capital=100000.0,
            strategy_type="breakout",
            user_id="test_multi_user"
        )
        self.assertIsNotNone(sbx2_id)
        
        sandboxes = database.get_sandboxes(user_id="test_multi_user")
        self.assertEqual(len(sandboxes), 2)
        
        fetched1 = database.get_sandbox_by_id(sbx1_id, user_id="test_multi_user")
        self.assertEqual(fetched1["name"], "RSI Mean Reversion")
        self.assertEqual(float(fetched1["initial_capital"]), 50000.0)
        self.assertEqual(float(fetched1["cash_balance"]), 50000.0)
        
        # Test fuzzy name lookup
        matched = database.get_sandbox_by_name("momentum", user_id="test_multi_user")
        self.assertIsNotNone(matched)
        self.assertEqual(str(matched["sandbox_id"]), str(sbx2_id))

    def test_max_10_sandboxes_limit_enforcement(self):
        # Create 10 sandboxes
        for i in range(10):
            database.create_sandbox(
                name=f"Sandbox #{i+1}",
                initial_capital=10000.0,
                user_id="test_multi_user"
            )
            
        sandboxes = database.get_sandboxes(user_id="test_multi_user")
        self.assertEqual(len(sandboxes), 10)
        
        # Attempting 11th should raise ValueError
        with self.assertRaises(ValueError):
            database.create_sandbox(
                name="Sandbox #11 (Overflow)",
                initial_capital=10000.0,
                user_id="test_multi_user"
            )

    def test_subledger_position_and_cash_isolation(self):
        sbx_a = database.create_sandbox(name="Sandbox A", initial_capital=100000.0, user_id="test_multi_user")
        sbx_b = database.create_sandbox(name="Sandbox B", initial_capital=50000.0, user_id="test_multi_user")
        
        # Buy 10 NVDA @ $120 in Sandbox A ($1,200 cost)
        database.update_sandbox_position_and_cash(
            sandbox_id=sbx_a,
            symbol="NVDA",
            qty=10.0,
            execution_price=120.0,
            side="buy"
        )
        
        # Verify Sandbox A updated
        pos_a = database.get_sandbox_positions(sbx_a)
        self.assertEqual(len(pos_a), 1)
        self.assertEqual(pos_a[0]["symbol"], "NVDA")
        self.assertEqual(float(pos_a[0]["qty"]), 10.0)
        self.assertEqual(float(pos_a[0]["avg_entry_price"]), 120.0)
        
        sbx_a_record = database.get_sandbox_by_id(sbx_a, user_id="test_multi_user")
        self.assertEqual(float(sbx_a_record["cash_balance"]), 100000.0 - 1200.0)
        
        # Verify Sandbox B remained completely unaffected
        pos_b = database.get_sandbox_positions(sbx_b)
        self.assertEqual(len(pos_b), 0)
        sbx_b_record = database.get_sandbox_by_id(sbx_b, user_id="test_multi_user")
        self.assertEqual(float(sbx_b_record["cash_balance"]), 50000.0)

    def test_reset_and_delete_sandbox(self):
        sbx_id = database.create_sandbox(name="Sandbox to Reset", initial_capital=25000.0, user_id="test_multi_user")
        
        # Add position
        database.update_sandbox_position_and_cash(
            sandbox_id=sbx_id,
            symbol="AAPL",
            qty=5.0,
            execution_price=200.0,
            side="buy"
        )
        self.assertEqual(len(database.get_sandbox_positions(sbx_id)), 1)
        
        # Reset
        success = database.reset_sandbox(sbx_id, user_id="test_multi_user")
        self.assertTrue(success)
        self.assertEqual(len(database.get_sandbox_positions(sbx_id)), 0)
        rec = database.get_sandbox_by_id(sbx_id, user_id="test_multi_user")
        self.assertEqual(float(rec["cash_balance"]), 25000.0)
        
        # Delete
        del_success = database.delete_sandbox(sbx_id, user_id="test_multi_user")
        self.assertTrue(del_success)
        self.assertIsNone(database.get_sandbox_by_id(sbx_id, user_id="test_multi_user"))

    def test_leaderboard_ranking(self):
        sbx1 = database.create_sandbox(name="Leader Sandbox", initial_capital=10000.0, user_id="test_multi_user")
        sbx2 = database.create_sandbox(name="Laggard Sandbox", initial_capital=10000.0, user_id="test_multi_user")
        
        # Mock positions with high price for sbx1 and low for sbx2
        database.update_sandbox_position_and_cash(sbx1, "NVDA", 10.0, 100.0, "buy")
        database.update_sandbox_position_and_cash(sbx2, "TSLA", 10.0, 300.0, "buy")
        
        leaderboard = database.get_all_sandboxes_leaderboard(user_id="test_multi_user")
        self.assertEqual(len(leaderboard), 2)
        self.assertEqual(leaderboard[0]["rank"], 1)
        self.assertEqual(leaderboard[1]["rank"], 2)

    def test_create_sandbox_tool(self):
        res_msg = create_sandbox_tool(
            name="AI Created Sandbox",
            initial_capital=75000.0,
            strategy_type="macd"
        )
        self.assertIn("Strategy Sandbox Created Successfully", res_msg)
        self.assertIn("AI Created Sandbox", res_msg)
        self.assertIn("$75,000.00", res_msg)
        
        # Cleanup
        sbx = database.get_sandbox_by_name("AI Created Sandbox")
        if sbx:
            database.delete_sandbox(str(sbx["sandbox_id"]))

    def test_router_sandbox_target_intent(self):
        output = agent.route_user_intent(
            user_prompt="Buy 15 shares of NVDA in my RSI Mean Reversion sandbox",
            has_uploaded_file=False
        )
        action_types = [a.action_type for a in output.actions]
        self.assertIn("paper_trade", action_types)
        trade_action = next(a for a in output.actions if a.action_type == "paper_trade")
        self.assertEqual(trade_action.ticker, "NVDA")
        self.assertEqual(trade_action.qty, 15.0)
        self.assertEqual(trade_action.side, "buy")
        self.assertIsNotNone(trade_action.sandbox_target)

    @patch("agent.orchestrator.generate_ai_response")
    def test_synthesize_chat_response_tools(self, mock_generate):
        mock_generate.return_value = "Synthesized response"
        resp = agent.synthesize_chat_response(
            user_prompt="Test prompt",
            results_summary="Test summary",
            strategies_str="No strategies"
        )
        self.assertEqual(resp, "Synthesized response")
        mock_generate.assert_called_once()
        tools_arg = mock_generate.call_args[1].get("tools")
        self.assertIsNone(tools_arg)

if __name__ == "__main__":
    unittest.main()

