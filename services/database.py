
import time
import threading
import yfinance as yf
import agent
from .vector_store import get_embedding_provider

_scheduler_lock = threading.Lock()
import os
import re
import json
import psycopg2
from psycopg2 import pool, extras
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import config

db_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((psycopg2.OperationalError, psycopg2.InterfaceError))
)


_pool = None
_scheduler_started = False
def get_pool():
    global _pool
    if _pool is None:
        try:
            # Configure threaded pool to be thread-safe for Streamlit
            _pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=config.COCKROACH_DATABASE_URL
            )
        except Exception as e:
            print(f"Error initializing database connection pool: {e}")
            raise e
    return _pool

def get_db_connection():
    return get_pool().getconn()

def release_db_connection(conn):
    if _pool and conn:
        _pool.putconn(conn)

@db_retry
def run_migrations():
    """Reads SQL scripts in the migrations folder and runs them sequentially."""
    print("Running database migrations...")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Create schema_version table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INT PRIMARY KEY,
                    applied_at TIMESTAMPTZ DEFAULT now()
                );
            """)
            conn.commit()

            # Find all SQL migrations
            migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")
            if not os.path.exists(migrations_dir):
                print("Migrations directory does not exist. Skipping.")
                return

            migration_files = sorted([
                f for f in os.listdir(migrations_dir)
                if f.endswith(".sql") and re.match(r"^\d+", f)
            ])

            for f in migration_files:
                version = int(re.match(r"^(\d+)", f).group(1))
                
                # Check if applied
                cur.execute("SELECT version FROM schema_version WHERE version = %s;", (version,))
                if cur.fetchone() is not None:
                    continue  # Already applied
                
                print(f"Applying migration: {f} (Version {version})...")
                sql_path = os.path.join(migrations_dir, f)
                with open(sql_path, "r", encoding="utf-8") as sql_file:
                    migration_sql = sql_file.read()
                
                # Execute migration
                cur.execute(migration_sql)
                # Record migration
                cur.execute("INSERT INTO schema_version (version) VALUES (%s);", (version,))
                conn.commit()
                print(f"Migration {f} applied successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def add_holding(ticker: str, shares: float, cost_basis: float):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_holdings (ticker, shares, cost_basis)
                VALUES (%s, %s, %s)
                RETURNING holding_id;
                """,
                (ticker.upper().strip(), shares, cost_basis)
            )
            holding_id = cur.fetchone()[0]
            conn.commit()
            return holding_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_holdings():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT holding_id, ticker, shares, cost_basis, created_at FROM user_holdings ORDER BY ticker ASC;")
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def remove_holding(holding_id: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_holdings WHERE holding_id = %s;", (holding_id,))
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def save_strategy(strategy_text: str, embedding: list[float]):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # We must convert list[float] to string format representation compatible with CockroachDB VECTOR: '[1.0, 2.0, ...]'
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            cur.execute(
                """
                INSERT INTO user_strategies (strategy_text, embedding)
                VALUES (%s, %s::VECTOR)
                RETURNING strategy_id;
                """,
                (strategy_text, embedding_str)
            )
            strategy_id = cur.fetchone()[0]
            conn.commit()
            return strategy_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_strategies():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT strategy_id, strategy_text, created_at FROM user_strategies ORDER BY created_at DESC;")
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def remove_strategy(strategy_id: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_strategies WHERE strategy_id = %s;", (strategy_id,))
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def update_strategy(strategy_id: str, strategy_text: str, embedding: list[float]):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            embedding_str = "[" + ",".join(map(str, embedding)) + "]"
            cur.execute(
                """
                UPDATE user_strategies
                SET strategy_text = %s, embedding = %s::VECTOR
                WHERE strategy_id = %s;
                """,
                (strategy_text, embedding_str, strategy_id)
            )
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def search_strategies_semantic(query_embedding: list[float], limit: int = 3):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            # Use <=> for cosine distance in CockroachDB vector ops
            cur.execute(
                """
                SELECT strategy_id, strategy_text, created_at, (embedding <=> %s::VECTOR) AS distance
                FROM user_strategies
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (embedding_str, limit)
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def save_document_chunks(doc_name: str, ticker: str, chunks: list[dict]):
    """
    Saves a batch of document chunks.
    Each chunk dict should contain: 'chunk_index', 'chunk_text', 'embedding', and 'chunk_metadata'.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for chunk in chunks:
                embedding_str = "[" + ",".join(map(str, chunk['embedding'])) + "]"
                metadata_str = json.dumps(chunk.get('chunk_metadata', {}))
                cur.execute(
                    """
                    INSERT INTO document_chunks (document_name, ticker, chunk_index, chunk_text, embedding, chunk_metadata)
                    VALUES (%s, %s, %s, %s, %s::VECTOR, %s::JSONB);
                    """,
                    (doc_name, ticker.upper().strip(), chunk['chunk_index'], chunk['chunk_text'], embedding_str, metadata_str)
                )
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def search_document_chunks_semantic(ticker: str, query_embedding: list[float], limit: int = 5):
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
            cur.execute(
                """
                SELECT chunk_id, document_name, chunk_index, chunk_text, chunk_metadata, (embedding <=> %s::VECTOR) AS distance
                FROM document_chunks
                WHERE ticker = %s
                ORDER BY distance ASC
                LIMIT %s;
                """,
                (embedding_str, ticker.upper().strip(), limit)
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def log_research_session(
    prompt_query: str,
    retrieved_news: str,
    vector_distance: float,
    bull_perspective: str,
    bear_perspective: str,
    generated_summary: str,
    session_metadata: dict = None
):
    if session_metadata is None:
        session_metadata = {}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            metadata_str = json.dumps(session_metadata)
            cur.execute(
                """
                INSERT INTO research_audit_logs (
                    prompt_query, retrieved_news, vector_distance, 
                    bull_perspective, bear_perspective, generated_summary, session_metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::JSONB)
                RETURNING log_id;
                """,
                (prompt_query, retrieved_news, vector_distance, bull_perspective, bear_perspective, generated_summary, metadata_str)
            )
            log_id = cur.fetchone()[0]
            conn.commit()
            return log_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_research_logs():
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT log_id, prompt_query, retrieved_news, vector_distance, 
                       bull_perspective, bear_perspective, generated_summary, session_metadata, created_at 
                FROM research_audit_logs 
                ORDER BY created_at DESC;
                """
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def save_stock_price(ticker: str, price: float, daily_change_pct: float):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO stock_prices (ticker, price, daily_change_pct, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (ticker)
                DO UPDATE SET price = EXCLUDED.price, daily_change_pct = EXCLUDED.daily_change_pct, updated_at = now();
                """,
                (ticker.upper().strip(), price, daily_change_pct)
            )
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_latest_prices() -> dict:
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT ticker, price, daily_change_pct, updated_at FROM stock_prices;")
            rows = cur.fetchall()
            return {
                row["ticker"]: {
                    "price": float(row["price"]),
                    "daily_change_pct": float(row["daily_change_pct"]) if row["daily_change_pct"] is not None else 0.0,
                    "updated_at": row["updated_at"]
                }
                for row in rows
            }
    finally:
        release_db_connection(conn)

@db_retry
def save_portfolio_snapshot(total_value: float, total_gain_loss: float, total_gain_loss_pct: float) -> str:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO portfolio_snapshots (total_value, total_gain_loss, total_gain_loss_pct)
                VALUES (%s, %s, %s)
                RETURNING snapshot_id;
                """,
                (total_value, total_gain_loss, total_gain_loss_pct)
            )
            snapshot_id = cur.fetchone()[0]
            conn.commit()
            return snapshot_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_portfolio_snapshots(limit: int = 100) -> list[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT snapshot_id, total_value, total_gain_loss, total_gain_loss_pct, recorded_at
                FROM portfolio_snapshots
                ORDER BY recorded_at ASC
                LIMIT %s;
                """,
                (limit,)
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def save_chat_message(role: str, content: str, user_id: str = 'demo_user') -> str:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chat_history (role, content, user_id)
                VALUES (%s, %s, %s)
                RETURNING message_id;
                """,
                (role, content, user_id)
            )
            message_id = cur.fetchone()[0]
            conn.commit()
            return message_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_chat_history(limit: int = 20, user_id: str = 'demo_user') -> list[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT role, content FROM (
                    SELECT role, content, created_at FROM chat_history
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                ) sub
                ORDER BY created_at ASC;
                """,
                (user_id, limit)
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def clear_chat_history(user_id: str = 'demo_user'):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_history WHERE user_id = %s;", (user_id,))
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def save_market_news(ticker: str, title: str, source: str, url: str, summary: str, published_at: str, user_id: str = 'demo_user') -> bool:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO market_news (user_id, ticker, title, source, url, summary, published_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ticker, title) DO NOTHING
                RETURNING news_id;
                """,
                (user_id, ticker.upper().strip(), title.strip(), source.strip(), url, summary, published_at)
            )
            inserted = cur.fetchone()
            conn.commit()
            return inserted is not None
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_market_news(ticker: str = None, limit: int = 30, user_id: str = 'demo_user') -> list[dict]:
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if ticker:
                cur.execute(
                    """
                    SELECT news_id, ticker, title, source, url, summary, published_at, action_suggestions, created_at
                    FROM market_news
                    WHERE user_id = %s AND ticker = %s
                    ORDER BY published_at DESC
                    LIMIT %s;
                    """,
                    (user_id, ticker.upper().strip(), limit)
                )
            else:
                cur.execute(
                    """
                    SELECT news_id, ticker, title, source, url, summary, published_at, action_suggestions, created_at
                    FROM market_news
                    WHERE user_id = %s
                    ORDER BY published_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit)
                )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def update_news_suggestions(news_id: str, suggestions: str):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE market_newsget_embedding_provider
                SET action_suggestions = %s
                WHERE news_id = %s;
                """,
                (suggestions, news_id)
            )
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_market_news_by_id(news_id: str) -> dict:
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT news_id, ticker, title, source, url, summary, published_at, action_suggestions, created_at
                FROM market_news
                WHERE news_id = %s;
                """,
                (news_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        release_db_connection(conn)

def add_stock_holding(ticker: str, shares: float, cost_basis: float) -> str:
    """Adds a holding to the database and converts ticker to uppercase."""
    return add_holding(ticker.upper().strip(), shares, cost_basis)

def get_stock_holdings() -> list[dict]:
    """Retrieves all active holdings."""
    return get_holdings()

def remove_stock_holding(holding_id: str):
    """Removes a holding by ID."""
    remove_holding(holding_id)

def save_investment_strategy(strategy_text: str) -> str:
    """Generates embedding for a strategy text and saves it to the """
    strategy_text = strategy_text.strip()
    if not strategy_text:
        raise ValueError("Strategy text cannot be empty.")
    
    print(f"Generating embedding for strategy rule: '{strategy_text[:40]}...'")
    embedding = ().get_embedding(strategy_text)
    return save_strategy(strategy_text, embedding)

def get_investment_strategies() -> list[dict]:
    """Retrieves all user strategies."""
    return get_strategies()

def update_strategy_by_reference(target: str, new_text: str) -> str:
    """
    Locates the strategy rule by target index/descriptor, generates embedding,
    and updates it in CockroachDB. Returns a confirmation message.
    """
    current_strats = get_strategies()
    matching_id = agent.resolve_strategy_match(target, current_strats)
    if not matching_id:
        raise ValueError(f"Could not find matching strategy for reference: '{target}'")
        
    embedding = get_embedding_provider().get_embedding(new_text)
    update_strategy(matching_id, new_text, embedding)
    
    matched_s = next(s for s in current_strats if s['strategy_id'] == matching_id)
    return matched_s['strategy_text']

def price_polling_loop():
    """
    Background daemon loop that fetches stock prices and writes snapshots to CockroachDB.
    """
    global _scheduler_started
    print("Background price polling loop started.")
    while True:
        try:
            holdings = get_holdings()
            if holdings:
                tickers = list(set(h["ticker"] for h in holdings))
                print(f"Polling prices for tickers: {tickers}")
                
                latest_prices = {}
                for t in tickers:
                    try:
                        ticker_obj = yf.Ticker(t)
                        info = ticker_obj.fast_info
                        price = float(info.get("lastPrice") or info.get("last_price") or 0.0)
                        
                        # Fallback to info dict
                        if price == 0.0:
                            price = float(ticker_obj.info.get("regularMarketPrice") or 0.0)
                            daily_change = float(ticker_obj.info.get("regularMarketChangePercent") or 0.0)
                        else:
                            daily_change = float(ticker_obj.info.get("regularMarketChangePercent") or 0.0)
                            
                        # Fallback to history close
                        if price == 0.0:
                            hist = ticker_obj.history(period="1d")
                            if not hist.empty:
                                price = float(hist["Close"].iloc[-1])
                                if "Open" in hist.columns and hist["Open"].iloc[-1] > 0:
                                    daily_change = ((price - hist["Open"].iloc[-1]) / hist["Open"].iloc[-1]) * 100
                        
                        if price > 0.0:
                            save_stock_price(t, price, daily_change)
                            latest_prices[t] = {"price": price, "daily_change_pct": daily_change}
                            print(f"  Saved price for {t}: ${price:.2f} ({daily_change:+.2f}%)")
                    except Exception as ex:
                        print(f"  Error polling price for {t}: {ex}")
                
                if latest_prices:
                    total_value = 0.0
                    total_cost = 0.0
                    for h in holdings:
                        ticker = h["ticker"]
                        shares = float(h["shares"])
                        cost_basis = float(h["cost_basis"])
                        
                        p_info = latest_prices.get(ticker)
                        if p_info:
                            current_price = p_info["price"]
                        else:
                            db_cache = get_latest_prices()
                            current_price = db_cache.get(ticker, {}).get("price", cost_basis)
                            
                        total_cost += shares * cost_basis
                        total_value += shares * current_price
                        
                    total_gain_loss = total_value - total_cost
                    total_gain_loss_pct = (total_gain_loss / total_cost) * 100 if total_cost > 0 else 0.0
                    
                    save_portfolio_snapshot(total_value, total_gain_loss, total_gain_loss_pct)
                    print(f"  [+] Logged snapshot: Value ${total_value:.2f}, Gain ${total_gain_loss:.2f} ({total_gain_loss_pct:+.2f}%)")
            else:
                print("Portfolio is empty. Skipping price polling.")
        except Exception as e:
            print(f"Error in price polling loop iteration: {e}")
            
        interval_sec = config.PRICE_POLLING_INTERVAL_MINUTES * 60
        if interval_sec <= 0:
            interval_sec = 1800
        time.sleep(interval_sec)

def initialize_polling_scheduler():
    """
    Safely spins up the background pricing daemon thread.
    Uses double-check locking to ensure only a single thread runs per app lifetime.
    """
    global _scheduler_started
    if not _scheduler_started:
        with _scheduler_lock:
            if not _scheduler_started:
                t = threading.Thread(target=price_polling_loop, daemon=True)
                t.start()
                _scheduler_started = True
                print("Real-time stock price scheduler thread spawned.")



import datetime
from .news_service import fetch_and_store_news, get_stored_news


def calculate_performance_metrics():
    holdings = get_holdings()
    latest_prices = get_latest_prices()
    total_value = 0.0
    total_cost = 0.0
    daily_change_val = 0.0
    prev_day_value_sum = 0.0
    holdings_details = []
    
    for h in holdings:
        ticker = h['ticker']
        shares = float(h['shares'])
        cost_basis = float(h['cost_basis'])
        
        p_data = latest_prices.get(ticker, {})
        current_price = float(p_data.get('price', 0.0))
        daily_change_pct = float(p_data.get('daily_change_pct', 0.0))
        
        position_cost = shares * cost_basis
        position_value = shares * current_price
        
        if daily_change_pct:
            prev_close = current_price / (1.0 + (daily_change_pct / 100.0))
            ticker_daily_change_val = (current_price - prev_close) * shares
        else:
            ticker_daily_change_val = 0.0
            prev_close = current_price
            
        gain_loss = position_value - position_cost
        gain_loss_pct = (gain_loss / position_cost * 100.0) if position_cost > 0 else 0.0
        
        total_cost += position_cost
        total_value += position_value
        daily_change_val += ticker_daily_change_val
        prev_day_value_sum += (prev_close * shares)
        
        holdings_details.append({
            'ticker': ticker,
            'shares': shares,
            'cost_basis': cost_basis,
            'current_price': current_price,
            'position_cost': position_cost,
            'position_value': position_value,
            'gain_loss': gain_loss,
            'gain_loss_pct': gain_loss_pct,
            'daily_change_pct': daily_change_pct
        })
        
    total_gain_loss = total_value - total_cost
    total_gain_loss_pct = (total_gain_loss / total_cost * 100.0) if total_cost > 0 else 0.0
    portfolio_daily_change_pct = (daily_change_val / prev_day_value_sum * 100.0) if prev_day_value_sum > 0 else 0.0
    
    return {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain_loss': total_gain_loss,
        'total_gain_loss_pct': total_gain_loss_pct,
        'daily_change': daily_change_val,
        'daily_change_pct': portfolio_daily_change_pct,
        'holdings_details': holdings_details
    }

def get_portfolio_performance_summary():
    metrics = calculate_performance_metrics()
    if metrics['total_cost'] == 0.0:
        return 'Your portfolio is currently empty. Add positions to calculate performance.'
        
    summary = f'### Portfolio Performance Metrics:\n- **Total Market Value**: ${metrics["total_value"]:,.2f}\n- **Total Portfolio Cost**: ${metrics["total_cost"]:,.2f}\n- **Total Gain/Loss**: ${metrics["total_gain_loss"]:,.2f} ({metrics["total_gain_loss_pct"]:+.2f}%)\n- **Daily Change**: ${metrics["daily_change"]:,.2f} ({metrics["daily_change_pct"]:+.2f}%)\n\n**Asset Allocations & Details:**\n'
    
    for h in metrics['holdings_details']:
        summary += f'- **{h["ticker"]}**: {h["shares"]} shares | Cost Basis: ${h["cost_basis"]:.2f} | Current Price: ${h["current_price"]:.2f} (Daily: {h["daily_change_pct"]:+.2f}%) | Value: ${h["position_value"]:.2f} (Gain/Loss: ${h["gain_loss"]:+.2f} / {h["gain_loss_pct"]:+.2f}%)\n'
        
    return summary

def execute_stress_test(scenario_prompt):
    start_time = time.time()
    print(f"Executing macro stress test for scenario: '{scenario_prompt}'...")
    holdings = get_holdings()
    
    holdings_str = ''
    for h in holdings:
        holdings_str += f"- {h['ticker']}: {h['shares']} shares @ ${h['cost_basis']}\n"
    if not holdings_str:
        holdings_str = "No stock holdings in portfolio."
        
    strategies = get_strategies()
    strategies_str = ''
    for s in strategies:
        strategies_str += f"{s['strategy_text']}\n"
    if not strategies_str:
        strategies_str = "No strategy rules configured."
        
    stress_report = agent.run_macro_stress_test(scenario_prompt, holdings_str, strategies_str)
    
    elapsed_time = round(time.time() - start_time, 2)
    session_metadata = {
        "execution_latency_sec": elapsed_time,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generative_model": config.GENERATIVE_MODEL,
        "type": "macro_stress_test"
    }
    
    log_research_session(
        prompt_query=f"Macro Stress Test: {scenario_prompt}",
        retrieved_news="N/A",
        vector_distance=1.0,
        bull_perspective=None,
        bear_perspective=None,
        generated_summary=stress_report,
        session_metadata=session_metadata
    )
    
    return stress_report

def conduct_portfolio_analysis(ticker):
    ticker = ticker.upper().strip()
    start_time = time.time()
    print(f"Starting research synthesis workflow for {ticker}...")
    
    holdings = get_holdings()
    ticker_holding = next((h for h in holdings if h['ticker'] == ticker), None)
    
    if ticker_holding:
        holdings_str = f"Ticker: {ticker}, Shares: {ticker_holding['shares']}, Cost Basis: ${ticker_holding['cost_basis']}"
    else:
        holdings_str = f"Ticker: {ticker} (No holdings in current portfolio)"
        
    full_holdings_context = 'Target Asset Holdings:\n' + holdings_str + '\n\nComplete Portfolio Allocation:\n'
    for h in holdings:
        full_holdings_context += f"- {h['ticker']}: {h['shares']} shares @ ${h['cost_basis']}\n"
    if not holdings:
        full_holdings_context += "No holdings"
        
    query_text = f"Investment analysis and risk assessment rules for {ticker} stock."
    
    strategies_context = "No strategy rules configured."
    try:
        from .vector_store import get_embedding_provider
        query_embed = get_embedding_provider().get_embedding(query_text)
        matching_strategies = search_strategies_semantic(query_embed, limit=5)
        if matching_strategies:
            strategies_context = ''
            for r in matching_strategies:
                strategies_context += f"- Rule (distance {r['distance']:.3f}): {r['strategy_text']}\n"
    except Exception as e:
        print(f"Warning: Semantic strategy search failed: {e}")
        
    news = "No news found for this ticker."
    try:
        fetch_and_store_news(ticker)
        n = get_stored_news(ticker, limit=5)
        if n:
            news = ''
            for item in n:
                news += f"- {item['title']} (Source: {item['source']}, Published: {item['published_at']})\n  Summary: {item['summary']}\n"
    except Exception as e:
        print(f"Warning: News fetch failed: {e}")
        
    docs_context = "No uploaded earnings transcripts or documents found for this ticker."
    try:
        matching_chunks = search_document_chunks_semantic(ticker, query_embed, limit=5)
        if matching_chunks:
            docs_context = ''
            for c in matching_chunks:
                docs_context += f"- Document: {c['document_name']} (Chunk {c['chunk_index']}, distance {c['distance']:.3f})\n{c['chunk_text']}\n"
    except Exception as e:
        print(f"Warning: Semantic document chunks search failed: {e}")
        
    print("Orchestrating parallel Bull vs. Bear debate...")
    debate_res = agent.run_parallel_debate(ticker, full_holdings_context, news, strategies_context, docs_context)
    
    print("Synthesizing debate cases...")
    synthesis = agent.synthesize_debate(ticker, debate_res['bull'], debate_res['bear'])
    
    elapsed_time = round(time.time() - start_time, 2)
    session_metadata = {
        "execution_latency_sec": elapsed_time,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "embedding_model": config.EMBEDDING_MODEL,
        "generative_model": config.GENERATIVE_MODEL,
        "news_sources_count": len(n) if 'n' in locals() and n else 0
    }
    
    print("Logging research session into CockroachDB...")
    log_research_session(
        prompt_query=f"Portfolio research scan for {ticker}",
        retrieved_news=news[:500] + '...' if len(news) > 500 else news,
        vector_distance=1.0,
        bull_perspective=debate_res['bull'],
        bear_perspective=debate_res['bear'],
        generated_summary=synthesis,
        session_metadata=session_metadata
    )
    
    print(f"Analysis completed successfully in {elapsed_time:.2f}s!")
    return synthesis
