
import time
import threading
from typing import Callable, Optional
import yfinance as yf
import agent

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
            migrations_dir = os.path.join(os.path.dirname(__file__), "..", "migrations")
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
    from .ticker_service import canonicalize_ticker
    canonical_symbol = canonicalize_ticker(ticker)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_holdings (ticker, shares, cost_basis)
                VALUES (%s, %s, %s)
                RETURNING holding_id;
                """,
                (canonical_symbol, shares, cost_basis)
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
                UPDATE market_news
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
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM market_news WHERE news_id = %s;", (news_id,))
            row = cur.fetchone()
            if not row:
                return {}
            colnames = [desc[0] for desc in cur.description]
            return dict(zip(colnames, row))
    finally:
        release_db_connection(conn)

def add_stock_holding(symbol: str, shares: float, avg_price: float) -> str:
    """Adds a holding and returns the new holding ID."""
    return add_holding(symbol, shares, avg_price)

def get_stock_holdings() -> list[dict]:
    """Retrieves all user holdings."""
    return get_holdings()

def remove_stock_holding(holding_id: str):
    """Removes a holding by ID."""
    remove_holding(holding_id)

def save_investment_strategy(strategy_text: str) -> str:
    """Generates embedding for a strategy text and saves it to the database."""
    from .vector_store import get_embedding_provider
    strategy_text = strategy_text.strip()
    if not strategy_text:
        raise ValueError("Strategy text cannot be empty.")
    
    print(f"Generating embedding for strategy rule: '{strategy_text[:40]}...'")
    embedding = get_embedding_provider().get_embedding(strategy_text)
    return save_strategy(strategy_text, embedding)

def get_investment_strategies() -> list[dict]:
    """Retrieves all user strategies."""
    return get_strategies()

def update_strategy_by_reference(target: str, new_text: str) -> str:
    """
    Locates the strategy rule by target index/descriptor, generates embedding,
    and updates it in CockroachDB. Returns a confirmation message.
    """
    from .vector_store import get_embedding_provider
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
                from .ticker_service import fetch_realtime_price
                tickers = list(set(h["ticker"] for h in holdings))
                print(f"Polling prices for tickers: {tickers}")
                
                latest_prices = {}
                for t in tickers:
                    try:
                        price, daily_change, canonical_t = fetch_realtime_price(t)
                        if price > 0.0:
                            save_stock_price(canonical_t, price, daily_change)
                            latest_prices[canonical_t] = {"price": price, "daily_change_pct": daily_change}
                            print(f"  Saved price for {canonical_t}: ${price:.2f} ({daily_change:+.2f}%)")
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
            'daily_change_pct': daily_change_pct,
            'created_at': h.get('created_at')
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

def get_portfolio_performance_summary(status_callback: Optional[Callable[[str, Optional[str]], None]] = None):
    if status_callback:
        status_callback("📈 Computing portfolio performance metrics...", "Fetching latest market quotes and calculating valuation & returns...")
    metrics = calculate_performance_metrics()
    if metrics['total_cost'] == 0.0:
        return 'Your portfolio is currently empty. Add positions to calculate performance.'
        
    summary = f'### Portfolio Performance Metrics:\n- **Total Market Value**: ${metrics["total_value"]:,.2f}\n- **Total Portfolio Cost**: ${metrics["total_cost"]:,.2f}\n- **Total Gain/Loss**: ${metrics["total_gain_loss"]:,.2f} ({metrics["total_gain_loss_pct"]:+.2f}%)\n- **Daily Change**: ${metrics["daily_change"]:,.2f} ({metrics["daily_change_pct"]:+.2f}%)\n\n**Asset Allocations & Details:**\n'
    
    for h in metrics['holdings_details']:
        summary += f'- **{h["ticker"]}**: {h["shares"]} shares | Cost Basis: ${h["cost_basis"]:.2f} | Current Price: ${h["current_price"]:.2f} (Daily: {h["daily_change_pct"]:+.2f}%) | Value: ${h["position_value"]:.2f} (Gain/Loss: ${h["gain_loss"]:+.2f} / {h["gain_loss_pct"]:+.2f}%)\n'
        
    return summary

def execute_stress_test(scenario_prompt, status_callback: Optional[Callable[[str, Optional[str]], None]] = None):
    start_time = time.time()
    print(f"Executing macro stress test for scenario: '{scenario_prompt}'...")
    if status_callback:
        status_callback("📊 Loading active portfolio holdings & strategy rules...", "Querying CockroachDB relational engine...")
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
        
    if status_callback:
        status_callback("⚡ Simulating macro shock impact via Gemini...", f"Evaluating scenario: '{scenario_prompt}' against portfolio assets...")
    stress_report = agent.run_macro_stress_test(scenario_prompt, holdings_str, strategies_str)
    
    elapsed_time = round(time.time() - start_time, 2)
    session_metadata = {
        "execution_latency_sec": elapsed_time,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generative_model": config.GENERATIVE_MODEL,
        "type": "macro_stress_test"
    }
    
    if status_callback:
        status_callback("💾 Saving stress test audit log...", "Logging report to CockroachDB with 30-day Row-Level TTL...")
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

def conduct_portfolio_analysis(ticker, status_callback: Optional[Callable[[str, Optional[str]], None]] = None):
    ticker = ticker.upper().strip()
    start_time = time.time()
    print(f"Starting research synthesis workflow for {ticker}...")
    
    if status_callback:
        status_callback(f"📊 Loading portfolio context for {ticker}...", "Querying user_holdings table from CockroachDB...")
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
        if status_callback:
            status_callback(f"🔍 Searching qualitative strategy vectors for {ticker}...", "Generating 768-dim embedding & querying CockroachDB HNSW index...")
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
        if status_callback:
            status_callback(f"🌐 Fetching real-time market news for {ticker}...", "Gathering latest headlines and financial catalyst updates...")
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
        if status_callback:
            status_callback(f"📄 Searching document transcripts for {ticker}...", "Running vector similarity search across document_chunks...")
        matching_chunks = search_document_chunks_semantic(ticker, query_embed, limit=5)
        if matching_chunks:
            docs_context = ''
            for c in matching_chunks:
                docs_context += f"- Document: {c['document_name']} (Chunk {c['chunk_index']}, distance {c['distance']:.3f})\n{c['chunk_text']}\n"
    except Exception as e:
        print(f"Warning: Semantic document chunks search failed: {e}")
        
    print("Orchestrating parallel Bull vs. Bear debate...")
    if status_callback:
        status_callback(f"⚔️ Dispatching Bull vs. Bear debate agents for {ticker}...", "Executing dual-agent reasoning for catalysts and downside risks...")
    debate_res = agent.run_parallel_debate(ticker, full_holdings_context, news, strategies_context, docs_context)
    
    print("Synthesizing debate cases...")
    if status_callback:
        status_callback(f"📝 Synthesizing debate cases for {ticker}...", "Evaluating bull/bear arguments against qualitative strategy guidelines...")
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
    if status_callback:
        status_callback(f"💾 Saving research audit log with Row-Level TTL...", "Persisting session to research_audit_logs table...")
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
    return {
        "bull": debate_res['bull'],
        "bear": debate_res['bear'],
        "synthesis": synthesis,
        "docs_context": docs_context
    }

# ==========================================
# PAPER TRADING AUDIT LOGGING
# ==========================================

@db_retry
def log_paper_trade(
    order_id: str,
    symbol: str,
    side: str,
    qty: float,
    status: str,
    execution_price: Optional[float] = None,
    order_type: str = "market",
    time_in_force: str = "gtc",
    raw_response: dict = None,
    sandbox_id: Optional[str] = None,
    user_id: str = "demo_user"
) -> str:
    """Logs an executed paper order into the paper_trades audit table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            raw_json = json.dumps(raw_response or {})
            cur.execute(
                """
                INSERT INTO paper_trades (
                    user_id, order_id, symbol, side, qty, 
                    execution_price, status, order_type, time_in_force, raw_response, sandbox_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::JSONB, %s)
                RETURNING trade_id;
                """,
                (
                    user_id,
                    str(order_id),
                    symbol.upper().strip(),
                    side.upper().strip(),
                    float(qty),
                    execution_price,
                    status,
                    order_type.lower().strip(),
                    time_in_force.lower().strip(),
                    raw_json,
                    sandbox_id
                )
            )
            trade_id = cur.fetchone()[0]
            conn.commit()
            return str(trade_id)
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_paper_trade_logs(limit: int = 50, sandbox_id: Optional[str] = None, user_id: str = "demo_user") -> list[dict]:
    """Retrieves executed paper trade audit logs ordered by creation timestamp descending."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if sandbox_id:
                cur.execute(
                    """
                    SELECT trade_id, user_id, order_id, symbol, side, qty, 
                           execution_price, status, order_type, time_in_force, raw_response, sandbox_id, created_at
                    FROM paper_trades
                    WHERE user_id = %s AND sandbox_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (user_id, sandbox_id, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT trade_id, user_id, order_id, symbol, side, qty, 
                           execution_price, status, order_type, time_in_force, raw_response, sandbox_id, created_at
                    FROM paper_trades
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (user_id, limit)
                )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

# ==========================================
# MULTI-SANDBOX MANAGEMENT & SUB-LEDGERS
# ==========================================

@db_retry
def create_sandbox(
    name: str,
    description: Optional[str] = None,
    initial_capital: float = 100000.0,
    strategy_id: Optional[str] = None,
    strategy_type: Optional[str] = None,
    user_id: str = "demo_user"
) -> str:
    """Creates a new strategy sandbox, enforcing a maximum limit of 10 sandboxes per user."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM sandboxes WHERE user_id = %s;", (user_id,))
            current_count = cur.fetchone()[0]
            if current_count >= 10:
                raise ValueError("Maximum limit of 10 strategy sandboxes reached. Please delete an existing sandbox first.")
                
            cur.execute(
                """
                INSERT INTO sandboxes (
                    user_id, name, description, strategy_id, strategy_type, initial_capital, cash_balance
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING sandbox_id;
                """,
                (
                    user_id,
                    name.strip(),
                    description.strip() if description else None,
                    strategy_id if strategy_id else None,
                    strategy_type.strip() if strategy_type else None,
                    float(initial_capital),
                    float(initial_capital)
                )
            )
            sandbox_id = cur.fetchone()[0]
            conn.commit()
            return str(sandbox_id)
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_sandboxes(user_id: str = "demo_user") -> list[dict]:
    """Retrieves all sandboxes for a user with bound strategy details."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT s.sandbox_id, s.user_id, s.name, s.description, s.strategy_id, 
                       s.strategy_type, s.initial_capital, s.cash_balance, s.created_at, s.updated_at,
                       us.strategy_text
                FROM sandboxes s
                LEFT JOIN user_strategies us ON s.strategy_id = us.strategy_id
                WHERE s.user_id = %s
                ORDER BY s.created_at ASC;
                """,
                (user_id,)
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def get_sandbox_by_id(sandbox_id: str, user_id: str = "demo_user") -> Optional[dict]:
    """Retrieves a specific sandbox by ID."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT s.sandbox_id, s.user_id, s.name, s.description, s.strategy_id, 
                       s.strategy_type, s.initial_capital, s.cash_balance, s.created_at, s.updated_at,
                       us.strategy_text
                FROM sandboxes s
                LEFT JOIN user_strategies us ON s.strategy_id = us.strategy_id
                WHERE s.sandbox_id = %s AND s.user_id = %s;
                """,
                (sandbox_id, user_id)
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        release_db_connection(conn)

@db_retry
def get_sandbox_by_name(name: str, user_id: str = "demo_user") -> Optional[dict]:
    """Retrieves a sandbox by name (case-insensitive substring or exact match)."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT s.sandbox_id, s.user_id, s.name, s.description, s.strategy_id, 
                       s.strategy_type, s.initial_capital, s.cash_balance, s.created_at, s.updated_at,
                       us.strategy_text
                FROM sandboxes s
                LEFT JOIN user_strategies us ON s.strategy_id = us.strategy_id
                WHERE s.user_id = %s AND LOWER(s.name) = LOWER(%s);
                """,
                (user_id, name.strip())
            )
            row = cur.fetchone()
            if row:
                return dict(row)
                
            cur.execute(
                """
                SELECT s.sandbox_id, s.user_id, s.name, s.description, s.strategy_id, 
                       s.strategy_type, s.initial_capital, s.cash_balance, s.created_at, s.updated_at,
                       us.strategy_text
                FROM sandboxes s
                LEFT JOIN user_strategies us ON s.strategy_id = us.strategy_id
                WHERE s.user_id = %s AND LOWER(s.name) LIKE LOWER(%s)
                ORDER BY s.created_at ASC
                LIMIT 1;
                """,
                (user_id, f"%{name.strip()}%")
            )
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        release_db_connection(conn)

@db_retry
def delete_sandbox(sandbox_id: str, user_id: str = "demo_user") -> bool:
    """Deletes a sandbox, cascading open positions and preserving paper trade logs."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM sandboxes WHERE sandbox_id = %s AND user_id = %s RETURNING sandbox_id;",
                (sandbox_id, user_id)
            )
            deleted = cur.fetchone()
            conn.commit()
            return deleted is not None
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def reset_sandbox(sandbox_id: str, user_id: str = "demo_user") -> bool:
    """Resets a sandbox to its initial capital, removing all open positions."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sandboxes
                SET cash_balance = initial_capital, updated_at = now()
                WHERE sandbox_id = %s AND user_id = %s
                RETURNING initial_capital;
                """,
                (sandbox_id, user_id)
            )
            row = cur.fetchone()
            if not row:
                return False
            cur.execute("DELETE FROM sandbox_positions WHERE sandbox_id = %s;", (sandbox_id,))
            conn.commit()
            return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

@db_retry
def get_sandbox_positions(sandbox_id: str) -> list[dict]:
    """Retrieves all open positions in a given sandbox."""
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT position_id, sandbox_id, symbol, qty, avg_entry_price, created_at, updated_at
                FROM sandbox_positions
                WHERE sandbox_id = %s
                ORDER BY symbol ASC;
                """,
                (sandbox_id,)
            )
            return [dict(row) for row in cur.fetchall()]
    finally:
        release_db_connection(conn)

@db_retry
def update_sandbox_position_and_cash(
    sandbox_id: str,
    symbol: str,
    qty: float,
    execution_price: float,
    side: str
):
    """
    Updates the sub-ledger for a specific sandbox (cash balance & positions table).
    """
    clean_symbol = symbol.upper().strip()
    clean_side = side.lower().strip()
    qty = float(qty)
    execution_price = float(execution_price)
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            trade_cost = qty * execution_price
            if clean_side == "buy":
                cur.execute(
                    """
                    UPDATE sandboxes
                    SET cash_balance = cash_balance - %s, updated_at = now()
                    WHERE sandbox_id = %s;
                    """,
                    (trade_cost, sandbox_id)
                )
                
                # Check existing position
                cur.execute(
                    "SELECT position_id, qty, avg_entry_price FROM sandbox_positions WHERE sandbox_id = %s AND symbol = %s;",
                    (sandbox_id, clean_symbol)
                )
                existing = cur.fetchone()
                if existing:
                    pos_id, cur_qty, cur_avg = existing
                    cur_qty = float(cur_qty)
                    cur_avg = float(cur_avg)
                    new_qty = cur_qty + qty
                    new_avg = ((cur_qty * cur_avg) + (qty * execution_price)) / new_qty
                    cur.execute(
                        """
                        UPDATE sandbox_positions
                        SET qty = %s, avg_entry_price = %s, updated_at = now()
                        WHERE position_id = %s;
                        """,
                        (new_qty, new_avg, pos_id)
                    )
                else:
                    cur.execute(
                        """
                        INSERT INTO sandbox_positions (sandbox_id, symbol, qty, avg_entry_price)
                        VALUES (%s, %s, %s, %s);
                        """,
                        (sandbox_id, clean_symbol, qty, execution_price)
                    )
            elif clean_side == "sell":
                cur.execute(
                    """
                    UPDATE sandboxes
                    SET cash_balance = cash_balance + %s, updated_at = now()
                    WHERE sandbox_id = %s;
                    """,
                    (trade_cost, sandbox_id)
                )
                cur.execute(
                    "SELECT position_id, qty FROM sandbox_positions WHERE sandbox_id = %s AND symbol = %s;",
                    (sandbox_id, clean_symbol)
                )
                existing = cur.fetchone()
                if existing:
                    pos_id, cur_qty = existing
                    cur_qty = float(cur_qty)
                    new_qty = cur_qty - qty
                    if new_qty <= 0.0001:
                        cur.execute("DELETE FROM sandbox_positions WHERE position_id = %s;", (pos_id,))
                    else:
                        cur.execute(
                            "UPDATE sandbox_positions SET qty = %s, updated_at = now() WHERE position_id = %s;",
                            (new_qty, pos_id)
                        )
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_db_connection(conn)

def calculate_sandbox_metrics(sandbox_id: str, user_id: str = "demo_user") -> dict:
    """
    Computes live portfolio metrics for a specific sandbox (Equity, Cash, Unrealized P&L, Total Return %).
    """
    sbx = get_sandbox_by_id(sandbox_id, user_id=user_id)
    if not sbx:
        return {}
        
    positions = get_sandbox_positions(sandbox_id)
    cash = float(sbx["cash_balance"])
    initial_capital = float(sbx["initial_capital"])
    
    positions_value = 0.0
    total_cost_basis = 0.0
    serialized_positions = []
    
    latest_prices = get_latest_prices()
    
    for p in positions:
        sym = p["symbol"]
        qty = float(p["qty"])
        avg_entry = float(p["avg_entry_price"])
        cost = qty * avg_entry
        total_cost_basis += cost
        
        cur_price = None
        if sym in latest_prices:
            cur_price = float(latest_prices[sym]["price"])
        if cur_price is None or cur_price <= 0:
            try:
                from .ticker_service import fetch_realtime_price
                p, _, _ = fetch_realtime_price(sym, fallback_price=avg_entry)
                cur_price = p if p > 0 else avg_entry
            except Exception:
                cur_price = avg_entry
        if cur_price is None or cur_price <= 0:
            cur_price = avg_entry
            
        mkt_val = qty * cur_price
        positions_value += mkt_val
        unrealized_pl = mkt_val - cost
        unrealized_plpc = (unrealized_pl / cost * 100.0) if cost > 0 else 0.0
        
        serialized_positions.append({
            "symbol": sym,
            "qty": qty,
            "avg_entry_price": avg_entry,
            "current_price": cur_price,
            "market_value": mkt_val,
            "cost_basis": cost,
            "unrealized_pl": unrealized_pl,
            "unrealized_plpc": unrealized_plpc,
            "side": "LONG"
        })
        
    total_equity = cash + positions_value
    total_pl = total_equity - initial_capital
    total_return_pct = (total_pl / initial_capital * 100.0) if initial_capital > 0 else 0.0
    
    return {
        "sandbox_id": str(sbx["sandbox_id"]),
        "name": sbx["name"],
        "description": sbx.get("description"),
        "strategy_id": str(sbx["strategy_id"]) if sbx.get("strategy_id") else None,
        "strategy_type": sbx.get("strategy_type"),
        "strategy_text": sbx.get("strategy_text"),
        "initial_capital": initial_capital,
        "cash": cash,
        "positions_value": positions_value,
        "equity": total_equity,
        "buying_power": max(0.0, cash),
        "total_pl": total_pl,
        "total_return_pct": total_return_pct,
        "positions_count": len(positions),
        "positions": serialized_positions
    }

def get_all_sandboxes_leaderboard(user_id: str = "demo_user") -> list[dict]:
    """Generates a ranked leaderboard of all sandboxes for a user."""
    sandboxes = get_sandboxes(user_id=user_id)
    leaderboard = []
    for s in sandboxes:
        m = calculate_sandbox_metrics(str(s["sandbox_id"]), user_id=user_id)
        if m:
            leaderboard.append(m)
            
    leaderboard.sort(key=lambda x: x.get("total_return_pct", 0.0), reverse=True)
    for idx, item in enumerate(leaderboard):
        item["rank"] = idx + 1
    return leaderboard

