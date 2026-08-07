import abc
import datetime
from google import genai
from google.genai import types
import yfinance as yf
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import config

# ==========================================
# RETRY CONFIGURATION
# ==========================================

# Retries API calls 3 times with exponential backoff.
# Catches general connection errors or rate limit errors.
api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)

# ==========================================
# EMBEDDING PROVIDER ABSTRACTION
# ==========================================

class EmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Generates a 768-dimensional embedding vector for the text."""
        pass

class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set in environment variables.")
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

    @api_retry
    def get_embedding(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )
        return response.embeddings[0].values

# ==========================================
# NEWS PROVIDER ABSTRACTION
# ==========================================

class NewsProvider(abc.ABC):
    @abc.abstractmethod
    def fetch_news(self, ticker: str) -> list[dict]:
        """
        Fetches latest news for a ticker.
        Returns a list of dicts with: ['title', 'source', 'url', 'summary', 'published_at']
        """
        pass

class YahooFinanceNewsProvider(NewsProvider):
    def fetch_news(self, ticker: str) -> list[dict]:
        try:
            print(f"Fetching news for {ticker} from Yahoo Finance...")
            stock = yf.Ticker(ticker)
            raw_news = stock.news
            
            if not raw_news:
                print(f"No news found for {ticker} on Yahoo Finance.")
                return []
                
            normalized_news = []
            for item in raw_news:
                # Check for new nested structure (item -> content -> ...)
                content = item.get("content")
                if isinstance(content, dict):
                    title = content.get("title") or "No Title"
                    
                    # Source extraction
                    provider = content.get("provider") or {}
                    source = provider.get("displayName") or "Yahoo Finance"
                    
                    # URL extraction
                    click_url = content.get("clickThroughUrl") or {}
                    canonical_url = content.get("canonicalUrl") or {}
                    url = click_url.get("url") or canonical_url.get("url") or ""
                    
                    # Summary extraction
                    summary = content.get("summary") or content.get("description") or title
                    
                    # Published Date extraction
                    pub_date = content.get("pubDate")
                    if not pub_date:
                        pub_date = datetime.datetime.now(datetime.timezone.utc).isoformat()
                else:
                    # Fallback to old top-level structure
                    title = item.get("title") or "No Title"
                    source = item.get("publisher") or "Yahoo Finance"
                    url = item.get("link") or ""
                    summary = item.get("summary") or item.get("title") or ""
                    
                    pub_time = item.get("providerPublishTime")
                    if pub_time:
                        pub_date = datetime.datetime.fromtimestamp(pub_time, datetime.timezone.utc).isoformat()
                    else:
                        pub_date = datetime.datetime.now(datetime.timezone.utc).isoformat()
                
                normalized_news.append({
                    "title": title.strip(),
                    "source": source.strip(),
                    "url": url,
                    "summary": summary.strip(),
                    "published_at": pub_date
                })
            return normalized_news
        except Exception as e:
            print(f"Error fetching from Yahoo Finance for {ticker}: {e}")
            return []

class TavilyNewsProvider(NewsProvider):
    def __init__(self):
        self.client = None
        if config.TAVILY_API_KEY:
            self.client = TavilyClient(api_key=config.TAVILY_API_KEY)

    @api_retry
    def fetch_news(self, ticker: str) -> list[dict]:
        if not self.client:
            print("Tavily API key is missing. Skipping search.")
            return []
            
        try:
            print(f"Fetching news for {ticker} from Tavily Search...")
            query = f"{ticker} stock latest news financial analysis"
            response = self.client.search(query=query, max_results=5)
            raw_results = response.get("results", [])
            
            normalized_news = []
            for item in raw_results:
                normalized_news.append({
                    "title": item.get("title", "No Title"),
                    "source": "Tavily Search API",
                    "url": item.get("url", ""),
                    "summary": item.get("content", ""),
                    "published_at": datetime.datetime.now(datetime.timezone.utc).isoformat() # Tavily doesn't guarantee published date
                })
            return normalized_news
        except Exception as e:
            print(f"Error fetching from Tavily for {ticker}: {e}")
            return []

class HybridNewsProvider(NewsProvider):
    def __init__(self):
        self.yfinance_provider = YahooFinanceNewsProvider()
        self.tavily_provider = TavilyNewsProvider()

    def fetch_news(self, ticker: str) -> list[dict]:
        # Try Yahoo Finance first
        news = self.yfinance_provider.fetch_news(ticker)
        if not news:
            print(f"Yahoo Finance returned no news for {ticker}. Falling back to Tavily...")
            news = self.tavily_provider.fetch_news(ticker)
        else:
            # If we get yfinance news, we can optionally supplement it with Tavily news for richer context
            if config.TAVILY_API_KEY:
                print(f"Supplementing news for {ticker} with Tavily Search...")
                tavily_news = self.tavily_provider.fetch_news(ticker)
                # Keep unique news items by title (case-insensitive simple check)
                existing_titles = {n['title'].lower().strip() for n in news}
                for item in tavily_news:
                    if item['title'].lower().strip() not in existing_titles:
                        news.append(item)
        return news[:6]  # Limit to top 6 news articles to avoid context bloat
