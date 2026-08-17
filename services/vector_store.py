import services.storage_service as storage
import time
import datetime
import threading
from typing import Callable, Optional
import yfinance as yf
from google.genai import errors
import services.database as database
import services.alpaca_service as alpaca_service
import providers
import agent
import config
_embedding_provider = None

def get_embedding_provider() -> providers.EmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = providers.GeminiEmbeddingProvider()
    return _embedding_provider

HELP_MESSAGE = """### ⚡ MarketPulse AI Slash Commands & Actions

You can trigger specific research sentinel actions using slash commands with optional natural language descriptions:

| Command | Action | 
| :--- | :--- | 
| **`/debate`** | **Bull vs. Bear Debate & News Scan** |
| **`/backtest`** | **Quantitative Strategy Backtest** |
| **`/trade`** | **Automated Paper Trading** |
| **`/stress`** | **Macro Risk Stress Test** |
| **`/help`** | **Commands Reference** |

💡 **Multi-Intent Support:** You can combine slash commands with other requests in a single prompt (e.g., `"/debate AAPL and also run a backtest on RSI"`).
"""

def run_chatbot_session(user_prompt: str, uploaded_files: list = None, file_context: str = None, status_callback: Optional[Callable[[str, Optional[str]], None]] = None, chat_history: list = None) -> dict:
    """
    Structured chatbot assistant workflow:
    1. Parses user's request into structured action items.
    2. Runs actions (debate, stress test, ingestion, strategy manipulation, paper trading) sequentially.
    3. Synthesizes outputs into a unified chat response.
    4. Logs session metadata and details in CockroachDB.
    """
    user_prompt = user_prompt.strip()
    
    # Instant response for /help
    if user_prompt.lower() in ["/help", "/commands", "help", "/help commands"]:
        if status_callback:
            status_callback("⚡ Displaying Slash Command Guide...", "Showing available slash commands and usage examples")
        return {
            "response": HELP_MESSAGE,
            "router": {"explanation": "Instant Slash Command Guide requested via /help.", "actions": []},
            "actions_run": [{"type": "help", "status": "success"}]
        }
        
    has_file = bool(uploaded_files)
    start_time = time.time()
    
    # Clear any previous turn's backtest and trade results
    agent.clear_last_backtest_result()
    agent.clear_last_trade_result()
    
    if status_callback:
        status_callback("🔍 Loading context & investment rules...", "Retrieving active portfolio holdings and qualitative strategy rules from CockroachDB...")
        
    # Pre-fetch strategies context to ensure chatbot is aware of active guidelines
    try:
        current_strats = database.get_strategies()
        strategies_str = "\n".join([f"- {s['strategy_text']}" for s in current_strats])
        if not strategies_str:
            strategies_str = "No active investment strategy guidelines configured."
    except Exception as e:
        print(f"Error fetching strategies for chatbot context: {e}")
        current_strats = []
        strategies_str = "Error fetching qualitative strategy guidelines."
        
    # Pre-fetch holdings context to assist router in picking ticker if unstated
    try:
        current_holdings = database.get_holdings()
        holdings_str = ", ".join([f"{h['ticker']} ({h['shares']} shares)" for h in current_holdings]) if current_holdings else "No holdings in portfolio."
    except Exception as e:
        print(f"Error fetching holdings for chatbot context: {e}")
        current_holdings = []
        holdings_str = None
    
    custom_holdings = database.extract_holdings_from_context(file_context)
    
    print(f"Routing chatbot user prompt: '{user_prompt[:50]}...'")
    router_output = agent.route_user_intent(user_prompt, has_uploaded_file=has_file, file_context=file_context, holdings_str=holdings_str, status_callback=status_callback, chat_history=chat_history)
    
    # Veto incorrect backtest routing on qualitative document/file queries
    has_tech_keywords = any(
        k in user_prompt.lower()
        for k in ["rsi", "macd", "sma", "ema", "moving average", "crossover", "bollinger", "breakout", "golden cross", "death cross", "rsi_period"]
    )
    is_ips_or_file_query = any(
        k in user_prompt.lower()
        for k in ["ips", "file", "document", "upload", "ips_sample", "txt", "pdf", "rules", "strategies"]
    )
    if is_ips_or_file_query and not has_tech_keywords and router_output and router_output.actions:
        filtered_actions = []
        for a in router_output.actions:
            if a.action_type.lower() == "backtest":
                print("Vetoing incorrect technical backtest action on qualitative document query.")
                continue
            filtered_actions.append(a)
        router_output.actions = filtered_actions

    results = []
    actions_run = []
    pending_strategy = None
    pending_portfolio_overwrite = None
    debate_payload = None
    
    for a in router_output.actions:
        a_type = a.action_type.lower()
        if a_type == "none":
            continue
            
        print(f"Executing routed action: {a_type}")
        
        if a_type == "debate":
            ticker = a.ticker.upper().strip() if a.ticker else None
            if not ticker:
                if custom_holdings:
                    ticker = custom_holdings[0]['ticker'].upper().strip()
                elif current_holdings:
                    ticker = current_holdings[0]['ticker'].upper().strip()
                
            if ticker:
                if status_callback:
                    status_callback(f"⚔️ Initiating research scan & Bull vs. Bear debate for {ticker}...", f"Scanning news and analyzing catalysts for {ticker}...")
                try:
                    res = database.conduct_portfolio_analysis(ticker, status_callback=status_callback)
                    synthesis_text = res.get('synthesis', str(res)) if isinstance(res, dict) else str(res)
                    results.append(
                        f"=== Portfolio News Scan & Debate for {ticker} ===\n"
                        f"{synthesis_text}\n"
                    )
                    actions_run.append({"type": "debate", "ticker": ticker, "status": "success"})
                    debate_payload = res.get("structured_debate")
                except Exception as e:
                    results.append(f"=== Debate for {ticker} Failed ===\nError: {e}\n")
                    actions_run.append({"type": "debate", "ticker": ticker, "status": "error", "error": str(e)})
                    if status_callback:
                        status_callback(f"⚠️ Debate for {ticker} encountered an issue", f"Error: {e}")
            else:
                results.append("=== Debate Error ===\nStock ticker symbol was not specified and no portfolio holdings were found.\n")
                actions_run.append({"type": "debate", "status": "missing_args"})
                
        elif a_type == "stress_test":
            scenario = a.scenario
            if scenario:
                if status_callback:
                    status_callback(f"⚡ Running macro stress test...", f"Evaluating scenario: '{scenario}'...")
                try:
                    report = database.execute_stress_test(scenario, custom_holdings=custom_holdings, status_callback=status_callback)
                    results.append(
                        f"=== Macro Stress Test Report ===\n"
                        f"Scenario: {scenario}\n"
                        f"{report}\n"
                    )
                    actions_run.append({"type": "stress_test", "scenario": scenario, "status": "success"})
                except Exception as e:
                    results.append(f"=== Stress Test Failed ===\nError: {e}\n")
                    actions_run.append({"type": "stress_test", "scenario": scenario, "status": "error", "error": str(e)})
                    if status_callback:
                        status_callback(f"⚠️ Stress test encountered an issue", f"Error: {e}")
            else:
                results.append("=== Stress Test Error ===\nScenario description was not specified by router.\n")
                actions_run.append({"type": "stress_test", "status": "missing_args"})
                
        elif a_type == "ingest":
            ticker = a.ticker.upper().strip() if a.ticker else None
            if not has_file:
                results.append(
                    f"=== Ingestion Failed ===\n"
                    f"No documents were uploaded to ingest.\n"
                )
                actions_run.append({"type": "ingest", "ticker": ticker, "status": "missing_file"})
            else:
                # If custom column overrides are provided, we run the CSV ingestion synchronously
                if getattr(a, "csv_ticker_col", None) or getattr(a, "csv_shares_col", None) or getattr(a, "csv_cost_col", None):
                    import services.storage_service as storage_service
                    file_name = uploaded_files[0]["name"]
                    file_data = uploaded_files[0]["bytes"]
                    try:
                        ingest_res = storage_service.ingest_portfolio_csv(
                            file_name=file_name,
                            file_data=file_data,
                            user_prompt=user_prompt,
                            ticker_col_override=a.csv_ticker_col,
                            shares_col_override=a.csv_shares_col,
                            cost_col_override=a.csv_cost_col
                        )
                        results.append(
                            f"=== CSV Portfolio Ingested with Custom Columns ===\n"
                            f"Successfully mapped columns:\n"
                            f"- Ticker: '{a.csv_ticker_col}'\n"
                            f"- Shares: '{a.csv_shares_col}'\n"
                            f"- Cost Basis: '{a.csv_cost_col}'\n"
                            f"Parsed {len(ingest_res['holdings'])} holdings.\n"
                        )
                        actions_run.append({"type": "ingest", "ticker": ticker, "status": "success"})
                        pending_portfolio_overwrite = {
                            "file_name": file_name,
                            "holdings": ingest_res["holdings"],
                            "overwrite_intent": ingest_res["overwrite_intent"]
                        }
                    except Exception as e:
                        results.append(
                            f"=== CSV Custom Ingestion Failed ===\n"
                            f"Error: {e}\n"
                        )
                        actions_run.append({"type": "ingest", "ticker": ticker, "status": "error", "error": str(e)})
                else:
                    # Since ingestion is completed asynchronously, we simply verify if the documents were processed.
                    results.append(
                        f"=== Ingestion Complete ===\n"
                        f"Documents have been successfully processed, embedded, and saved in the CockroachDB vector space.\n"
                    )
                    actions_run.append({"type": "ingest", "ticker": ticker, "status": "success"})
                


        elif a_type == "backtest":
            ticker = a.ticker.upper().strip() if a.ticker else None
            if not ticker:
                if custom_holdings:
                    ticker = custom_holdings[0]['ticker'].upper().strip()
                elif current_holdings:
                    ticker = current_holdings[0]['ticker'].upper().strip()
                
            if ticker:
                strat_type = a.strategy_type or "sma_cross"
                timeframe = a.timeframe or "1y"
                short_w = a.short_window or 20
                long_w = a.long_window or 50
                rsi_p = a.rsi_period or 14
                rsi_os = a.rsi_oversold if a.rsi_oversold is not None else 30.0
                rsi_ob = a.rsi_overbought if a.rsi_overbought is not None else 70.0
                m_fast = a.macd_fast or 12
                m_slow = a.macd_slow or 26
                m_sig = a.macd_signal or 9
                bb_w = a.bb_window or 20
                bb_s = a.bb_std if a.bb_std is not None else 2.0
                bo_w = a.breakout_window or 20
                
                if status_callback:
                    status_callback(f"📊 Running quantitative {strat_type.upper()} backtest for {ticker}...", f"Evaluating strategy logic on historical data...")
                try:
                    bt_res = agent.backtest_universal_strategy(
                        ticker=ticker,
                        strategy_type=strat_type,
                        period=timeframe,
                        short_window=short_w,
                        long_window=long_w,
                        rsi_period=rsi_p,
                        rsi_oversold=rsi_os,
                        rsi_overbought=rsi_ob,
                        macd_fast=m_fast,
                        macd_slow=m_slow,
                        macd_signal=m_sig,
                        bb_window=bb_w,
                        bb_std=bb_s,
                        breakout_window=bo_w
                    )
                    if bt_res.get("error"):
                        results.append(f"=== Backtest Error for {ticker} ({strat_type}) ===\n{bt_res['error']}\n")
                        actions_run.append({"type": "backtest", "ticker": ticker, "status": "error", "error": bt_res["error"]})
                    else:
                        outperform_str = "Outperformed Benchmark" if bt_res["outperformed"] else "Underperformed Benchmark"
                        results.append(
                            f"=== Quantitative Backtest for {ticker} ({bt_res['strategy_name']}) ===\n"
                            f"- Strategy: {bt_res['strategy_name']}\n"
                            f"- Rules: {bt_res['condition_summary']}\n"
                            f"- Strategy Total Return: {bt_res['Strategy_Return_Pct']:+.2f}%\n"
                            f"- Buy & Hold Benchmark Return: {bt_res['Buy_Hold_Return_Pct']:+.2f}%\n"
                            f"- Win Rate: {bt_res['Win_Rate_Pct']:.2f}%\n"
                            f"- Max Drawdown: {bt_res['Max_Drawdown_Pct']:.2f}%\n"
                            f"- Historical Outcome: {outperform_str}\n"
                        )
                        actions_run.append({"type": "backtest", "ticker": ticker, "strategy_type": strat_type, "status": "success"})
                except Exception as e:
                    results.append(f"=== Backtest for {ticker} Failed ===\nError: {e}\n")
                    actions_run.append({"type": "backtest", "ticker": ticker, "status": "error", "error": str(e)})
                    if status_callback:
                        status_callback(f"⚠️ Backtest for {ticker} failed", f"Error: {e}")
            else:
                results.append("=== Backtest Error ===\nTicker symbol was not specified and no portfolio holdings were found.\n")
                actions_run.append({"type": "backtest", "status": "missing_args"})

        elif a_type == "paper_trade":
            ticker = a.ticker.upper().strip() if a.ticker else None
            if not ticker:
                if custom_holdings:
                    ticker = custom_holdings[0]['ticker'].upper().strip()
                elif current_holdings:
                    ticker = current_holdings[0]['ticker'].upper().strip()
                
            qty = a.qty if a.qty is not None and a.qty > 0 else 10.0
            side = a.side.lower().strip() if a.side else "buy"
            if side not in ("buy", "sell"):
                side = "buy"
                
            target_sbx_name = getattr(a, "sandbox_target", None)
            
            # Check sandboxes
            all_sandboxes = database.get_sandboxes()
            target_sandbox_id = None
            resolved_sbx_name = None
            
            if not all_sandboxes:
                results.append(
                    f"=== Paper Trade Notice ===\n"
                    f"User attempted to trade {ticker}, but 0 strategy sandboxes exist. "
                    f"Offer to create a new dedicated sandbox for this strategy (e.g. '{ticker} Strategy Sandbox' with $100,000 capital).\n"
                )
                actions_run.append({"type": "paper_trade", "ticker": ticker, "status": "no_sandboxes"})
            elif ticker:
                if target_sbx_name:
                    matched = next((s for s in all_sandboxes if str(s["sandbox_id"]) == str(target_sbx_name)), None)
                    if not matched:
                        matched = next((s for s in all_sandboxes if target_sbx_name.lower() in s["name"].lower() or s["name"].lower() in target_sbx_name.lower()), None)
                    if matched:
                        target_sandbox_id = str(matched["sandbox_id"])
                        resolved_sbx_name = matched["name"]
                        
                if not target_sandbox_id and all_sandboxes:
                    target_sandbox_id = str(all_sandboxes[0]["sandbox_id"])
                    resolved_sbx_name = all_sandboxes[0]["name"]
                    
                if status_callback:
                    status_callback(f"🧪 Executing paper trade order for {qty:g} shares of {ticker}...", f"Targeting {resolved_sbx_name or 'Sandbox'} ({side.upper()} order)...")
                try:
                    trade_res = alpaca_service.submit_paper_order(
                        symbol=ticker,
                        qty=qty,
                        side=side,
                        sandbox_id=target_sandbox_id,
                        order_type="market",
                        time_in_force="gtc"
                    )
                    if resolved_sbx_name:
                        trade_res["sandbox_name"] = resolved_sbx_name
                    agent.set_last_trade_result(trade_res)
                    results.append(
                        f"=== Paper Trade Order Executed ===\n"
                        f"- Sandbox: {resolved_sbx_name or 'Default'}\n"
                        f"- Symbol: {trade_res['symbol']}\n"
                        f"- Action: {trade_res['side']}\n"
                        f"- Quantity: {trade_res['qty']}\n"
                        f"- Order ID: {trade_res['order_id']}\n"
                        f"- Status: {trade_res['status']}\n"
                        f"- Timestamp: {trade_res['timestamp']}\n"
                    )
                    actions_run.append({"type": "paper_trade", "ticker": ticker, "qty": qty, "side": side, "sandbox": resolved_sbx_name, "status": "success"})
                    if status_callback:
                        status_callback(f"✅ Paper order placed in {resolved_sbx_name or 'Sandbox'}", f"{side.upper()} {qty:g} {ticker}")
                except Exception as e:
                    results.append(f"=== Paper Trade for {ticker} Failed ===\nError: {e}\n")
                    actions_run.append({"type": "paper_trade", "ticker": ticker, "status": "error", "error": str(e)})
                    if status_callback:
                        status_callback(f"⚠️ Paper trade for {ticker} failed", f"Error: {e}")
            else:
                results.append("=== Paper Trade Error ===\nTicker symbol was not specified and no portfolio holdings were found.\n")
                actions_run.append({"type": "paper_trade", "status": "missing_args"})




        elif a_type == "list_strategies":
            if status_callback:
                status_callback("📋 Listing configured qualitative strategies...", "Querying strategy guidelines from CockroachDB...")
            try:
                if current_strats:
                    str_list = "\n".join([f"{idx+1}. {s['strategy_text']}" for idx, s in enumerate(current_strats)])
                    results.append(
                        f"=== Current Qualitative Strategy Rules ===\n"
                        f"{str_list}\n"
                    )
                else:
                    results.append("=== Current Qualitative Strategy Rules ===\nNo investment strategy rules are currently configured.\n")
                actions_run.append({"type": "list_strategies", "status": "success"})
            except Exception as e:
                results.append(f"=== List Strategies Failed ===\nError: {e}\n")
                actions_run.append({"type": "list_strategies", "status": "error", "error": str(e)})
                
    results_summary = "\n".join(results)
    if not results_summary:
        print("Executing conversational direct response...")
        final_response = agent.generate_conversational_response(
            user_prompt,
            strategies_str,
            file_context,
            status_callback=status_callback,
            allowed_actions=[a.action_type for a in router_output.actions],
            contract=router_output.contract,
            chat_history=chat_history
        )
        actions_run.append({"type": "conversational"})
    else:
        print("Synthesizing workflow actions results into chatbot response...")
        final_response = agent.synthesize_chat_response(
            user_prompt,
            results_summary,
            strategies_str,
            file_context,
            status_callback=status_callback,
            allowed_actions=[a.action_type for a in router_output.actions],
            contract=router_output.contract,
            chat_history=chat_history
        )
        
    elapsed = time.time() - start_time
    session_metadata = {
        "execution_latency_sec": round(elapsed, 2),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generative_model": config.GENERATIVE_MODEL,
        "type": "chatbot_conversation",
        "router_explanation": router_output.explanation,
        "actions_run": actions_run
    }
    
    database.log_research_session(
        prompt_query=f"Chatbot: {user_prompt}",
        retrieved_news="Chatbot synthesized results",
        vector_distance=1.0,
        bull_perspective=None,
        bear_perspective=None,
        generated_summary=final_response,
        session_metadata=session_metadata
    )
    
    bt_payload = agent.get_last_backtest_result()
    trade_payload = agent.get_last_trade_result()
    
    return {
        "response": final_response,
        "router": {
            "explanation": router_output.explanation,
            "actions": [{"type": a.action_type, "ticker": a.ticker, "scenario": a.scenario} for a in router_output.actions]
        },
        "actions_run": actions_run,
        "pending_strategy": pending_strategy,
        "pending_portfolio_overwrite": pending_portfolio_overwrite,
        "backtest_data": bt_payload if (bt_payload and not bt_payload.get("error")) else None,
        "trade_data": trade_payload if (trade_payload and not trade_payload.get("error")) else None,
        "debate_data": debate_payload
    }

def run_remaining_actions(remaining_actions: list, original_prompt: str, status_callback: Optional[Callable[[str, Optional[str]], None]] = None) -> dict:
    """
    Executes a list of remaining actions sequentially and returns the final synthesized response and metadata.
    Factors in the newly updated database state.
    """
    start_time = time.time()
    
    # Clear any previous turn's backtest and trade results
    agent.clear_last_backtest_result()
    agent.clear_last_trade_result()
    
    if status_callback:
        status_callback("🔄 Resuming remaining workflow actions...", f"Executing {len(remaining_actions)} queued action(s)...")
        
    try:
        current_strats = database.get_strategies()
        strategies_str = "\n".join([f"- {s['strategy_text']}" for s in current_strats])
        if not strategies_str:
            strategies_str = "No active investment strategy guidelines configured."
    except Exception as e:
        print(f"Error fetching strategies for chatbot context: {e}")
        current_strats = []
        strategies_str = "Error fetching qualitative strategy guidelines."
        
    results = []
    actions_run = []
    debate_payload = None
    
    for a_dict in remaining_actions:
        a_type = a_dict.get("action_type") if isinstance(a_dict, dict) else a_dict.action_type
        a_type = a_type.lower()
        
        print(f"Executing remaining continuation action: {a_type}")
        
        if a_type == "debate":
            ticker = a_dict.get("ticker") if isinstance(a_dict, dict) else a_dict.ticker
            ticker = ticker.upper().strip() if ticker else None
            if ticker:
                if status_callback:
                    status_callback(f"⚔️ Running Bull vs. Bear debate for {ticker}...", f"Scanning news and analyzing catalysts for {ticker}...")
                try:
                    res = database.conduct_portfolio_analysis(ticker, status_callback=status_callback)
                    synthesis_text = res.get('synthesis', str(res)) if isinstance(res, dict) else str(res)
                    results.append(
                        f"=== Portfolio News Scan & Debate for {ticker} ===\n"
                        f"{synthesis_text}\n"
                    )
                    actions_run.append({"type": "debate", "ticker": ticker, "status": "success"})
                    debate_payload = res.get("structured_debate")
                except Exception as e:
                    results.append(f"=== Debate for {ticker} Failed ===\nError: {e}\n")
                    actions_run.append({"type": "debate", "ticker": ticker, "status": "error", "error": str(e)})
                    if status_callback:
                        status_callback(f"⚠️ Debate for {ticker} failed", f"Error: {e}")
            else:
                results.append("=== Debate Error ===\nTicker symbol was not specified.\n")
                actions_run.append({"type": "debate", "status": "missing_args"})
                
        elif a_type == "stress_test":
            scenario = a_dict.get("scenario") if isinstance(a_dict, dict) else a_dict.scenario
            if scenario:
                if status_callback:
                    status_callback(f"⚡ Running macro stress test...", f"Simulating economic shock: '{scenario}'...")
                try:
                    report = database.execute_stress_test(scenario, status_callback=status_callback)
                    results.append(
                        f"=== Macro Stress Test Report ===\n"
                        f"Scenario: {scenario}\n"
                        f"{report}\n"
                    )
                    actions_run.append({"type": "stress_test", "scenario": scenario, "status": "success"})
                except Exception as e:
                    results.append(f"=== Stress Test Failed ===\nError: {e}\n")
                    actions_run.append({"type": "stress_test", "scenario": scenario, "status": "error", "error": str(e)})
                    if status_callback:
                        status_callback("⚠️ Stress test failed", f"Error: {e}")
            else:
                results.append("=== Stress Test Error ===\nScenario description was not specified.\n")
                actions_run.append({"type": "stress_test", "status": "missing_args"})
                
        elif a_type == "ingest":
            results.append("=== Ingestion Error ===\nDocument ingestion continuation is not supported from cached actions.\n")
            actions_run.append({"type": "ingest", "status": "unsupported_continuation"})
            


        elif a_type == "paper_trade":
            ticker = a_dict.get("ticker") if isinstance(a_dict, dict) else getattr(a_dict, "ticker", None)
            ticker = ticker.upper().strip() if ticker else None
            qty = a_dict.get("qty") if isinstance(a_dict, dict) else getattr(a_dict, "qty", 10.0)
            qty = float(qty) if qty and float(qty) > 0 else 10.0
            side = a_dict.get("side", "buy") if isinstance(a_dict, dict) else getattr(a_dict, "side", "buy")
            side = str(side).lower().strip()
            if side not in ("buy", "sell"):
                side = "buy"
                
            target_sbx_name = a_dict.get("sandbox_target") if isinstance(a_dict, dict) else getattr(a_dict, "sandbox_target", None)
            all_sandboxes = database.get_sandboxes()
            target_sandbox_id = None
            resolved_sbx_name = None
            
            if target_sbx_name and all_sandboxes:
                matched = next((s for s in all_sandboxes if str(s["sandbox_id"]) == str(target_sbx_name)), None)
                if not matched:
                    matched = next((s for s in all_sandboxes if target_sbx_name.lower() in s["name"].lower() or s["name"].lower() in target_sbx_name.lower()), None)
                if matched:
                    target_sandbox_id = str(matched["sandbox_id"])
                    resolved_sbx_name = matched["name"]
                    
            if not target_sandbox_id and all_sandboxes:
                target_sandbox_id = str(all_sandboxes[0]["sandbox_id"])
                resolved_sbx_name = all_sandboxes[0]["name"]
                
            if ticker and all_sandboxes:
                if status_callback:
                    status_callback(f"🧪 Executing paper trade order for {qty:g} shares of {ticker}...", f"Targeting {resolved_sbx_name or 'Sandbox'} ({side.upper()} order)...")
                try:
                    trade_res = alpaca_service.submit_paper_order(
                        symbol=ticker,
                        qty=qty,
                        side=side,
                        sandbox_id=target_sandbox_id,
                        order_type="market",
                        time_in_force="gtc"
                    )
                    if resolved_sbx_name:
                        trade_res["sandbox_name"] = resolved_sbx_name
                    agent.set_last_trade_result(trade_res)
                    results.append(
                        f"=== Paper Trade Order Executed ===\n"
                        f"- Sandbox: {resolved_sbx_name or 'Default'}\n"
                        f"- Symbol: {trade_res['symbol']}\n"
                        f"- Action: {trade_res['side']}\n"
                        f"- Quantity: {trade_res['qty']}\n"
                        f"- Order ID: {trade_res['order_id']}\n"
                        f"- Status: {trade_res['status']}\n"
                        f"- Timestamp: {trade_res['timestamp']}\n"
                    )
                    actions_run.append({"type": "paper_trade", "ticker": ticker, "qty": qty, "side": side, "sandbox": resolved_sbx_name, "status": "success"})
                except Exception as e:
                    results.append(f"=== Paper Trade for {ticker} Failed ===\nError: {e}\n")
                    actions_run.append({"type": "paper_trade", "ticker": ticker, "status": "error", "error": str(e)})
            else:
                results.append("=== Paper Trade Error ===\nTicker symbol was not specified or no sandboxes exist.\n")
                actions_run.append({"type": "paper_trade", "status": "missing_args"})


        elif a_type == "backtest":
            ticker = a_dict.get("ticker") if isinstance(a_dict, dict) else getattr(a_dict, "ticker", None)
            ticker = ticker.upper().strip() if ticker else None
            if ticker:
                strat_type = a_dict.get("strategy_type", "sma_cross") if isinstance(a_dict, dict) else getattr(a_dict, "strategy_type", "sma_cross")
                timeframe = a_dict.get("timeframe", "1y") if isinstance(a_dict, dict) else getattr(a_dict, "timeframe", "1y")
                short_w = a_dict.get("short_window", 20) if isinstance(a_dict, dict) else getattr(a_dict, "short_window", 20)
                long_w = a_dict.get("long_window", 50) if isinstance(a_dict, dict) else getattr(a_dict, "long_window", 50)
                rsi_p = a_dict.get("rsi_period", 14) if isinstance(a_dict, dict) else getattr(a_dict, "rsi_period", 14)
                rsi_os = a_dict.get("rsi_oversold", 30.0) if isinstance(a_dict, dict) else getattr(a_dict, "rsi_oversold", 30.0)
                rsi_ob = a_dict.get("rsi_overbought", 70.0) if isinstance(a_dict, dict) else getattr(a_dict, "rsi_overbought", 70.0)
                m_fast = a_dict.get("macd_fast", 12) if isinstance(a_dict, dict) else getattr(a_dict, "macd_fast", 12)
                m_slow = a_dict.get("macd_slow", 26) if isinstance(a_dict, dict) else getattr(a_dict, "macd_slow", 26)
                m_sig = a_dict.get("macd_signal", 9) if isinstance(a_dict, dict) else getattr(a_dict, "macd_signal", 9)
                bb_w = a_dict.get("bb_window", 20) if isinstance(a_dict, dict) else getattr(a_dict, "bb_window", 20)
                bb_s = a_dict.get("bb_std", 2.0) if isinstance(a_dict, dict) else getattr(a_dict, "bb_std", 2.0)
                bo_w = a_dict.get("breakout_window", 20) if isinstance(a_dict, dict) else getattr(a_dict, "breakout_window", 20)
                
                if status_callback:
                    status_callback(f"📊 Running quantitative {strat_type.upper()} backtest for {ticker}...", f"Evaluating strategy logic...")
                try:
                    bt_res = agent.backtest_universal_strategy(
                        ticker=ticker,
                        strategy_type=strat_type,
                        period=timeframe,
                        short_window=short_w,
                        long_window=long_w,
                        rsi_period=rsi_p,
                        rsi_oversold=rsi_os,
                        rsi_overbought=rsi_ob,
                        macd_fast=m_fast,
                        macd_slow=m_slow,
                        macd_signal=m_sig,
                        bb_window=bb_w,
                        bb_std=bb_s,
                        breakout_window=bo_w
                    )
                    if bt_res.get("error"):
                        results.append(f"=== Backtest Error for {ticker} ({strat_type}) ===\n{bt_res['error']}\n")
                        actions_run.append({"type": "backtest", "ticker": ticker, "status": "error", "error": bt_res["error"]})
                    else:
                        outperform_str = "Outperformed Benchmark" if bt_res["outperformed"] else "Underperformed Benchmark"
                        results.append(
                            f"=== Quantitative Backtest for {ticker} ({bt_res['strategy_name']}) ===\n"
                            f"- Strategy: {bt_res['strategy_name']}\n"
                            f"- Rules: {bt_res['condition_summary']}\n"
                            f"- Strategy Total Return: {bt_res['Strategy_Return_Pct']:+.2f}%\n"
                            f"- Buy & Hold Benchmark Return: {bt_res['Buy_Hold_Return_Pct']:+.2f}%\n"
                            f"- Win Rate: {bt_res['Win_Rate_Pct']:.2f}%\n"
                            f"- Max Drawdown: {bt_res['Max_Drawdown_Pct']:.2f}%\n"
                            f"- Historical Outcome: {outperform_str}\n"
                        )
                        actions_run.append({"type": "backtest", "ticker": ticker, "strategy_type": strat_type, "status": "success"})
                except Exception as e:
                    results.append(f"=== Backtest for {ticker} Failed ===\nError: {e}\n")
                    actions_run.append({"type": "backtest", "ticker": ticker, "status": "error", "error": str(e)})
                    if status_callback:
                        status_callback(f"⚠️ Backtest for {ticker} failed", f"Error: {e}")
            else:
                results.append("=== Backtest Error ===\nTicker symbol was not specified.\n")
                actions_run.append({"type": "backtest", "status": "missing_args"})
                
    results_summary = "\n".join(results)
    if not results_summary:
        print("Executing conversational continuation response...")
        final_response = agent.generate_conversational_response(original_prompt, strategies_str, status_callback=status_callback)
        actions_run.append({"type": "conversational"})
    else:
        print("Synthesizing workflow continuation results into chatbot response...")
        final_response = agent.synthesize_chat_response(original_prompt, results_summary, strategies_str, status_callback=status_callback)
        
    elapsed = time.time() - start_time
    session_metadata = {
        "execution_latency_sec": round(elapsed, 2),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "generative_model": config.GENERATIVE_MODEL,
        "type": "chatbot_conversation_continuation",
        "actions_run": actions_run
    }
    
    database.log_research_session(
        prompt_query=f"Chatbot Continuation: {original_prompt}",
        retrieved_news="Chatbot continuation synthesized results",
        vector_distance=1.0,
        bull_perspective=None,
        bear_perspective=None,
        generated_summary=final_response,
        session_metadata=session_metadata
    )
    
    bt_payload = agent.get_last_backtest_result()
    trade_payload = agent.get_last_trade_result()
    
    return {
        "response": final_response,
        "actions_run": actions_run,
        "backtest_data": bt_payload if (bt_payload and not bt_payload.get("error")) else None,
        "trade_data": trade_payload if (trade_payload and not trade_payload.get("error")) else None,
        "debate_data": debate_payload
    }


