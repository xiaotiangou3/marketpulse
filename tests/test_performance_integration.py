import sys
import database
import services
import providers
import agent

# ==========================================
# RUNTIME MOCK INJECTIONS (Bypasses AI API)
# ==========================================

class MockEmbeddingProvider(providers.EmbeddingProvider):
    def get_embedding(self, text: str) -> list[float]:
        return [0.05] * 768

services._embedding_provider = MockEmbeddingProvider()

# Mock route_user_intent to resolve performance analysis
def mock_route_user_intent(user_prompt: str, has_uploaded_file: bool) -> agent.RouterOutput:
    user_prompt_lower = user_prompt.lower()
    actions = []
    
    if "how is my portfolio" in user_prompt_lower or "gains" in user_prompt_lower or "performance" in user_prompt_lower:
        actions.append(agent.ActionItem(action_type="performance_analysis"))
    else:
        actions.append(agent.ActionItem(action_type="none"))
        
    return agent.RouterOutput(explanation="Mock routing for performance testing", actions=actions)

agent.route_user_intent = mock_route_user_intent

# Mock synthesis response
def mock_synthesize_chat_response(user_prompt: str, results_summary: str, *args, **kwargs) -> str:
    return f"Synthesized Response:\n{results_summary}"

agent.synthesize_chat_response = mock_synthesize_chat_response

def main():
    print("==================================================")
    print("   MARKETPULSE PERFORMANCE & METRICS TEST SUITE   ")
    print("   (Deterministic Math & Log Database Validation)")
    print("==================================================")
    
    try:
        # Step 1: Run migrations (verifies schema works)
        print("\n[Step 1] Running database migrations...")
        database.run_migrations()
        print("  [+] Migrations completed successfully.")
        
        # Clean up database holdings for clean test slate
        print("  Cleaning database tables to isolate test run...")
        conn = database.get_db_connection()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_holdings;")
            cur.execute("DELETE FROM stock_prices;")
            cur.execute("DELETE FROM portfolio_snapshots;")
        conn.commit()
        database.release_db_connection(conn)
        
        # Step 2: Clear any old records of MSFT/AAPL in stock_prices
        # By setting holdings, we control the test.
        print("\n[Step 2] Adding test holdings...")
        msft_id = services.add_stock_holding("MSFT", 10.0, 300.0) # Cost = $3000
        aapl_id = services.add_stock_holding("AAPL", 20.0, 150.0) # Cost = $3000
        # Total cost = $6000
        print(f"  [+] Added MSFT position (10 shares @ $300) -> ID: {msft_id}")
        print(f"  [+] Added AAPL position (20 shares @ $150) -> ID: {aapl_id}")
        
        # Step 3: Mock cached stock prices in database
        print("\n[Step 3] Mocking cached stock prices in DB...")
        # MSFT current price = $330 (+10% change from previous close)
        # AAPL current price = $165 (+10% change from previous close)
        database.save_stock_price("MSFT", 330.0, 10.0)
        database.save_stock_price("AAPL", 165.0, 10.0)
        print("  [+] Saved mock prices to stock_prices table.")
        
        # Step 4: Verify performance calculations
        print("\n[Step 4] Calculating portfolio performance metrics...")
        metrics = services.calculate_performance_metrics()
        
        # Calculations:
        # MSFT cost = 3000, value = 3300. Gain = +300. Change: prev_close = 330/1.1 = 300. Daily Change = 10 * 30 = 300.
        # AAPL cost = 3000, value = 3300. Gain = +300. Change: prev_close = 165/1.1 = 150. Daily Change = 20 * 15 = 300.
        # Total Value = $6600
        # Total Cost = $6000
        # Total Gain = +$600 (+10%)
        # Daily Change = +$600 (+10%)
        
        print(f"  - Total Portfolio Cost: ${metrics['total_cost']:.2f} (Expected: $6000.00)")
        print(f"  - Total Portfolio Value: ${metrics['total_value']:.2f} (Expected: $6600.00)")
        print(f"  - Total Gain/Loss: ${metrics['total_gain_loss']:.2f} (Expected: $600.00)")
        print(f"  - Total Gain/Loss %: {metrics['total_gain_loss_pct']:.2f}% (Expected: 10.00%)")
        print(f"  - Daily Change: ${metrics['daily_change']:.2f} (Expected: $600.00)")
        print(f"  - Daily Change %: {metrics['daily_change_pct']:.2f}% (Expected: 10.00%)")
        
        assert metrics['total_cost'] == 6000.0
        assert metrics['total_value'] == 6600.0
        assert metrics['total_gain_loss'] == 600.0
        assert metrics['total_gain_loss_pct'] == 10.0
        assert round(metrics['daily_change'], 2) == 600.0
        assert round(metrics['daily_change_pct'], 2) == 10.0
        print("  [+] Performance calculation math is 100% CORRECT!")
        
        # Step 5: Log Snapshot
        print("\n[Step 5] Triggering portfolio snapshot logging...")
        snap_id = database.save_portfolio_snapshot(
            metrics['total_value'],
            metrics['total_gain_loss'],
            metrics['total_gain_loss_pct']
        )
        print(f"  [+] Saved snapshot -> ID: {snap_id}")
        
        # Retrieve snapshots
        snapshots = database.get_portfolio_snapshots(limit=5)
        print(f"  [+] Found {len(snapshots)} snapshots in history.")
        assert len(snapshots) >= 1
        assert float(snapshots[-1]['total_value']) == 6600.0
        
        # Step 6: Test chatbot routing for performance analysis
        print("\n[Step 6] Testing chatbot intent route for: 'how is my portfolio doing today?'...")
        res = services.run_chatbot_session("how is my portfolio doing today?")
        print(f"  [+] Resolved Actions: {res['router']['actions']}")
        print(f"  [+] Assistant Response Summary:\n{res['response']}")
        assert res['actions_run'][0]['type'] == "performance_analysis"
        assert "Total Market Value**: $6,600.00" in res['response']
        assert "Total Gain/Loss**: $600.00 (+10.00%)" in res['response']
        
        # Step 7: Clean up
        print("\n[Step 7] Cleaning up test holdings...")
        services.remove_stock_holding(msft_id)
        services.remove_stock_holding(aapl_id)
        print("  [+] Clean up completed.")
        
        print("\n==================================================")
        print("   PERFORMANCE TEST COMPLETED SUCCESSFULLY!       ")
        print("==================================================")
        
    except Exception as e:
        print(f"\n[-] Performance test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
