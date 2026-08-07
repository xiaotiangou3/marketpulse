import time
import datetime
import threading
import yfinance as yf
from google.genai import errors
import services.database as database
import providers
import agent
import config
import services.storage_service as storage
_news_provider = None

def get_news_provider() -> providers.NewsProvider:
    global _news_provider
    if _news_provider is None:
        _news_provider = providers.HybridNewsProvider()
    return _news_provider

def fetch_and_store_news(ticker: str) -> int:
    """
    Fetches latest news for a ticker from the news provider and stores it.
    Returns the number of news articles newly saved in CockroachDB.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        return 0
    
    print(f"Fetching news for {ticker} to store in database...")
    news_items = get_news_provider().fetch_news(ticker)
    
    newly_saved_count = 0
    for item in news_items:
        try:
            # save_market_news returns True if it was a new record (not duplicated)
            is_new = database.save_market_news(
                ticker=ticker,
                title=item.get("title", "No Title"),
                source=item.get("source", "Unknown"),
                url=item.get("url", ""),
                summary=item.get("summary", ""),
                published_at=item.get("published_at")
            )
            if is_new:
                newly_saved_count += 1
        except Exception as e:
            print(f"Error saving news item '{item.get('title')[:30]}': {e}")
            
    return newly_saved_count

def get_stored_news(ticker: str = None, limit: int = 30) -> list[dict]:
    """
    Retrieves stored news articles from the database.
    """
    if ticker:
        ticker = ticker.upper().strip()
    return database.get_market_news(ticker=ticker, limit=limit)

def generate_suggestions_for_news(news_id: str) -> str:
    """
    Generates AI action suggestions for a specific news item and caches them.
    """
    news_item = database.get_market_news_by_id(news_id)
    if not news_item:
        raise ValueError(f"News item not found with ID: {news_id}")
        
    ticker = news_item["ticker"].upper().strip()
    
    # 1. Fetch holdings context
    holdings = database.get_holdings()
    ticker_holding = next((h for h in holdings if h['ticker'] == ticker), None)
    if ticker_holding:
        holdings_str = f"Ticker: {ticker}, Shares: {ticker_holding['shares']}, Cost Basis: ${ticker_holding['cost_basis']}"
    else:
        holdings_str = f"Ticker: {ticker} (No holdings in current portfolio)"
        
    portfolio_summary = "\n".join([f"- {h['ticker']}: {h['shares']} shares @ ${h['cost_basis']}" for h in holdings])
    full_holdings_context = f"Target Asset Holdings:\n{holdings_str}\n\nComplete Portfolio Allocation:\n{portfolio_summary if portfolio_summary else 'No holdings'}"
    
    # 2. Fetch active qualitative strategies
    strategies = database.get_strategies()
    strategies_str = "\n".join([f"- {s['strategy_text']}" for s in strategies])
    if not strategies_str:
        strategies_str = "No active investment strategy guidelines configured."
        
    # 3. Call agent to generate suggestions
    suggestions = agent.generate_news_suggestions(
        news_title=news_item["title"],
        news_summary=news_item["summary"] or news_item["title"],
        holdings_str=full_holdings_context,
        strategies_str=strategies_str
    )
    
    # 4. Cache suggestions in DB
    database.update_news_suggestions(news_id, suggestions)
    return suggestions

