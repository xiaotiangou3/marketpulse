import sys
import os
import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import services.database as database
import services
import providers
import agent

# ==========================================
# RUNTIME MOCK INJECTIONS (Bypasses AI API)
# ==========================================

class MockEmbeddingProvider(providers.EmbeddingProvider):
    def get_embedding(self, text: str) -> list[float]:
        return [0.05] * 768

class MockNewsProvider(providers.NewsProvider):
    def fetch_news(self, ticker: str) -> list[dict]:
        return [
            {
                "title": f"Test News for {ticker} - Accelerating Demand",
                "source": "Mock Financial News",
                "url": "http://example.com/test-news",
                "summary": f"This is a mock summary of Tesla stock news detailing strong growth in Q3.",
                "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            },
            {
                "title": f"Test News for {ticker} - Macro Risk Warning",
                "source": "Mock Financial News",
                "url": "http://example.com/test-news-2",
                "summary": f"Interest rate hikes are affecting industry demand for {ticker}.",
                "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        ]

def main():
    # Inject mocks for standalone run
    import services.news_service as news_svc
    news_svc._news_provider = MockNewsProvider()
    news_svc.get_news_provider = lambda: MockNewsProvider()
    services._embedding_provider = MockEmbeddingProvider()
    services.get_news_provider = lambda: MockNewsProvider()

    # Mock AI suggestions helper in agent
    agent.generate_news_suggestions = lambda news_title, news_summary, holdings_str, strategies_str: (
        f"### Mock AI suggestions for: {news_title}\n\n"
        f"- **Strategic Impact**: Ticker details show user holdings align with current growth.\n"
        f"- **Suggested Action**: Monitor strategies context closely."
    )

    print("==================================================")
    print("   MARKETPULSE AI NEWS INTEGRATION TEST SUITE     ")
    print("==================================================")
    
    try:
        # Step 1: Run migrations to apply 0004 migration
        print("\n[Step 1] Running database migrations...")
        database.run_migrations()
        print("  [+] Migrations completed successfully.")
        
        # Step 2: Set up test holdings and strategies
        print("\n[Step 2] Setting up holdings and strategy rules...")
        tsla_id = services.add_stock_holding("TSLA", 15.0, 180.0)
        strat_id = services.save_investment_strategy("Limit tech volatility exposure where possible.")
        print(f"  [+] Added TSLA holding -> ID: {tsla_id}")
        print(f"  [+] Saved strategy rule -> ID: {strat_id}")
        
        # Step 3: Fetch and store news for TSLA
        print("\n[Step 3] Fetching and storing news for TSLA...")
        inserted_count = services.fetch_and_store_news("TSLA")
        print(f"  [+] Inserted {inserted_count} new articles.")
        assert inserted_count == 2, f"Expected 2 inserted, got {inserted_count}"
        
        # Step 4: Verify duplicate prevention on second fetch
        print("\n[Step 4] Re-fetching news for TSLA to test duplicate prevention...")
        second_inserted_count = services.fetch_and_store_news("TSLA")
        print(f"  [+] Inserted on second fetch: {second_inserted_count} new articles.")
        assert second_inserted_count == 0, f"Expected 0 inserted, got {second_inserted_count}"
        
        # Step 5: Retrieve stored news
        print("\n[Step 5] Retrieving stored news for TSLA...")
        news_items = services.get_stored_news("TSLA")
        assert len(news_items) >= 2, f"Expected at least 2 articles, got {len(news_items)}"
        
        test_item = news_items[0]
        print(f"  [+] Retrieved article: '{test_item['title']}' from source '{test_item['source']}'")
        assert test_item["ticker"] == "TSLA"
        assert test_item["action_suggestions"] is None, "Expected no initial suggestions"
        
        # Step 6: Generate action suggestions
        print("\n[Step 6] Generating action suggestions for news item...")
        news_id = test_item["news_id"]
        suggestions = services.generate_suggestions_for_news(news_id)
        print(f"  [+] Suggestions generated:\n{suggestions}")
        assert "Mock AI suggestions" in suggestions
        
        # Step 7: Verify cached suggestions
        print("\n[Step 7] Verifying suggestions cache...")
        updated_news_items = services.get_stored_news("TSLA")
        updated_test_item = next(item for item in updated_news_items if item["news_id"] == news_id)
        assert updated_test_item["action_suggestions"] == suggestions, "Suggestions were not cached correctly"
        print("  [+] Verified suggestions were successfully cached.")
        
        # Step 8: Clean up
        print("\n[Step 8] Cleaning up holdings...")
        services.remove_stock_holding(tsla_id)
        database.remove_strategy(strat_id)
        print("  [+] Clean up completed.")
        
        print("\n==================================================")
        print("   NEWS INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        print("==================================================")
        
    except Exception as e:
        print(f"\n[-] News integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
