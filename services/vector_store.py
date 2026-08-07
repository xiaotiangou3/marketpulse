import services.storage_service as storage
import time
import datetime
import threading
import yfinance as yf
from google.genai import errors
import services.database as database
import providers
import agent
import config
_embedding_provider = None

def get_embedding_provider() -> providers.EmbeddingProvider:
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = providers.GeminiEmbeddingProvider()
    return _embedding_provider

def run_chatbot_session(user_prompt: str, uploaded_files: list = None, file_context: str = None) -> dict:
    """
    Structured chatbot assistant workflow:
    1. Parses user's request into structured action items.
    2. Runs actions (debate, stress test, ingestion, strategy manipulation) sequentially.
    3. Synthesizes outputs into a unified chat response.
    4. Logs session metadata and details in CockroachDB.
    """
    user_prompt = user_prompt.strip()
    has_file = bool(uploaded_files)
    start_time = time.time()
    
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
    
    print(f"Routing chatbot user prompt: '{user_prompt[:50]}...'")
    router_output = agent.route_user_intent(user_prompt, has_uploaded_file=has_file, file_context=file_context)
    
    results = []
    actions_run = []
    pending_strategy = None
    
    for a in router_output.actions:
        a_type = a.action_type.lower()
        if a_type == "none":
            continue
            
        print(f"Executing routed action: {a_type}")
        
        if a_type == "debate":
            ticker = a.ticker.upper().strip() if a.ticker else None
            if ticker:
                try:
                    res = database.conduct_portfolio_analysis(ticker)
                    results.append(
                        f"=== Portfolio News Scan & Debate for {ticker} ===\n"
                        f"{res['synthesis']}\n"
                    )
                    actions_run.append({"type": "debate", "ticker": ticker, "status": "success"})
                except Exception as e:
                    results.append(f"=== Debate for {ticker} Failed ===\nError: {e}\n")
                    actions_run.append({"type": "debate", "ticker": ticker, "status": "error", "error": str(e)})
            else:
                results.append("=== Debate Error ===\nTicker symbol was not specified by router.\n")
                actions_run.append({"type": "debate", "status": "missing_args"})
                
        elif a_type == "stress_test":
            scenario = a.scenario
            if scenario:
                try:
                    report = database.execute_stress_test(scenario)
                    results.append(
                        f"=== Macro Stress Test Report ===\n"
                        f"Scenario: {scenario}\n"
                        f"{report}\n"
                    )
                    actions_run.append({"type": "stress_test", "scenario": scenario, "status": "success"})
                except Exception as e:
                    results.append(f"=== Stress Test Failed ===\nError: {e}\n")
                    actions_run.append({"type": "stress_test", "scenario": scenario, "status": "error", "error": str(e)})
            else:
                results.append("=== Stress Test Error ===\nScenario description was not specified by router.\n")
                actions_run.append({"type": "stress_test", "status": "missing_args"})
                
        elif a_type == "ingest":
            ticker = a.ticker.upper().strip() if a.ticker else None
            if ticker and uploaded_files:
                success_count = 0
                for f_info in uploaded_files:
                    if f_info["name"].lower().endswith(".pdf"):
                        try:
                            storage.ingest_pdf_transcript(f_info["name"], f_info["bytes"], ticker)
                            results.append(
                                f"=== PDF Transcript Ingested ===\n"
                                f"Uploaded and indexed transcript '{f_info['name']}' for ticker {ticker} in CockroachDB vector space.\n"
                            )
                            actions_run.append({"type": "ingest", "ticker": ticker, "file": f_info["name"], "status": "success"})
                            success_count += 1
                        except Exception as e:
                            results.append(f"=== Ingestion Failed ({f_info['name']}) ===\nError: {e}\n")
                            actions_run.append({"type": "ingest", "ticker": ticker, "file": f_info["name"], "status": "error", "error": str(e)})
                if success_count == 0:
                    results.append("=== Ingestion Warning ===\nNo PDF files were uploaded to ingest. Non-PDF files were provided as context.\n")
            elif not uploaded_files:
                results.append(
                    "=== Ingestion Error ===\n"
                    "The user requested document ingestion, but no file was uploaded. "
                    "Instruct the user to attach a PDF file to their message.\n"
                )
                actions_run.append({"type": "ingest", "ticker": ticker, "status": "missing_file"})
            else:
                results.append("=== Ingestion Error ===\nTicker symbol was not specified for ingestion.\n")
                actions_run.append({"type": "ingest", "status": "missing_ticker"})
                
        elif a_type == "performance_analysis":
            try:
                perf_report = database.get_portfolio_performance_summary()
                results.append(perf_report)
                actions_run.append({"type": "performance_analysis", "status": "success"})
            except Exception as e:
                results.append(f"=== Performance Analysis Failed ===\nError: {e}\n")
                actions_run.append({"type": "performance_analysis", "status": "error", "error": str(e)})

        elif a_type == "add_strategy":
            strategy_text = a.strategy_text
            if strategy_text:
                remaining_actions = []
                current_action_idx = router_output.actions.index(a)
                for rem_a in router_output.actions[current_action_idx+1:]:
                    if rem_a.action_type.lower() != "none":
                        remaining_actions.append({
                            "action_type": rem_a.action_type,
                            "ticker": rem_a.ticker,
                            "scenario": rem_a.scenario,
                            "strategy_text": rem_a.strategy_text,
                            "strategy_target": rem_a.strategy_target
                        })
                pending_strategy = {
                    "action_type": "add_strategy",
                    "strategy_text": strategy_text,
                    "strategy_target": None,
                    "remaining_actions": remaining_actions,
                    "original_prompt": user_prompt
                }
                results.append(
                    f"=== Strategy Rule Drafted ===\n"
                    f"Drafted new strategy rule: '{strategy_text}'\n"
                    f"Remaining actions to run after user confirmation: {[act['action_type'] for act in remaining_actions]}\n"
                )
                actions_run.append({"type": "add_strategy", "strategy_text": strategy_text, "status": "drafted"})
                break
            else:
                results.append("=== Add Strategy Error ===\nStrategy guideline text was not specified by router.\n")
                actions_run.append({"type": "add_strategy", "status": "missing_args"})

        elif a_type == "delete_strategy":
            target = a.strategy_target
            if target:
                try:
                    matching_id = agent.resolve_strategy_match(target, current_strats)
                    if matching_id:
                        matched_s = next(s for s in current_strats if s['strategy_id'] == matching_id)
                        database.remove_strategy(matching_id)
                        results.append(
                            f"=== Strategy Rule Deleted ===\n"
                            f"Successfully removed strategy guideline: '{matched_s['strategy_text']}'\n"
                        )
                        actions_run.append({"type": "delete_strategy", "strategy_target": target, "status": "success"})
                    else:
                        results.append(
                            f"=== Delete Strategy Warning ===\n"
                            f"Could not find a matching strategy rule for reference: '{target}'\n"
                        )
                        actions_run.append({"type": "delete_strategy", "strategy_target": target, "status": "no_match"})
                except Exception as e:
                    results.append(f"=== Delete Strategy Failed ===\nError: {e}\n")
                    actions_run.append({"type": "delete_strategy", "strategy_target": target, "status": "error", "error": str(e)})
            else:
                results.append("=== Delete Strategy Error ===\nTarget strategy descriptor or index was not specified by router.\n")
                actions_run.append({"type": "delete_strategy", "status": "missing_args"})

        elif a_type == "update_strategy":
            target = a.strategy_target
            new_text = a.strategy_text
            if target:
                try:
                    matching_id = agent.resolve_strategy_match(target, current_strats)
                    if matching_id:
                        matched_s = next(s for s in current_strats if s['strategy_id'] == matching_id)
                        
                        # Default to existing text if not provided in user prompt
                        if not new_text:
                            new_text = matched_s['strategy_text']
                            
                        remaining_actions = []
                        current_action_idx = router_output.actions.index(a)
                        for rem_a in router_output.actions[current_action_idx+1:]:
                            if rem_a.action_type.lower() != "none":
                                remaining_actions.append({
                                    "action_type": rem_a.action_type,
                                    "ticker": rem_a.ticker,
                                    "scenario": rem_a.scenario,
                                    "strategy_text": rem_a.strategy_text,
                                    "strategy_target": rem_a.strategy_target
                                })
                        pending_strategy = {
                            "action_type": "update_strategy",
                            "strategy_text": new_text,
                            "strategy_target": target,
                            "remaining_actions": remaining_actions,
                            "original_prompt": user_prompt
                        }
                        results.append(
                            f"=== Strategy Rule Update Drafted ===\n"
                            f"Drafted update for strategy rule:\n"
                            f"From: '{matched_s['strategy_text']}'\n"
                            f"To: '{new_text}'\n"
                            f"Remaining actions to run after user confirmation: {[act['action_type'] for act in remaining_actions]}\n"
                        )
                        actions_run.append({"type": "update_strategy", "strategy_target": target, "new_text": new_text, "status": "drafted"})
                        break
                    else:
                        results.append(
                            f"=== Update Strategy Warning ===\n"
                            f"Could not find a matching strategy rule to update for reference: '{target}'\n"
                        )
                        actions_run.append({"type": "update_strategy", "strategy_target": target, "status": "no_match"})
                except Exception as e:
                    results.append(f"=== Update Strategy Failed ===\nError: {e}\n")
                    actions_run.append({"type": "update_strategy", "strategy_target": target, "status": "error", "error": str(e)})
            else:
                results.append("=== Update Strategy Error ===\nTarget strategy descriptor or index must be specified.\n")
                actions_run.append({"type": "update_strategy", "status": "missing_args"})

        elif a_type == "list_strategies":
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
        final_response = agent.generate_conversational_response(user_prompt, strategies_str, file_context)
        actions_run.append({"type": "conversational"})
    else:
        print("Synthesizing workflow actions results into chatbot response...")
        final_response = agent.synthesize_chat_response(user_prompt, results_summary, strategies_str, file_context)
        
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
    
    return {
        "response": final_response,
        "router": {
            "explanation": router_output.explanation,
            "actions": [{"type": a.action_type, "ticker": a.ticker, "scenario": a.scenario} for a in router_output.actions]
        },
        "actions_run": actions_run,
        "pending_strategy": pending_strategy
    }

def run_remaining_actions(remaining_actions: list, original_prompt: str) -> dict:
    """
    Executes a list of remaining actions sequentially and returns the final synthesized response and metadata.
    Factors in the newly updated database state.
    """
    start_time = time.time()
    
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
    
    for a_dict in remaining_actions:
        a_type = a_dict.get("action_type") if isinstance(a_dict, dict) else a_dict.action_type
        a_type = a_type.lower()
        
        print(f"Executing remaining continuation action: {a_type}")
        
        if a_type == "debate":
            ticker = a_dict.get("ticker") if isinstance(a_dict, dict) else a_dict.ticker
            ticker = ticker.upper().strip() if ticker else None
            if ticker:
                try:
                    res = database.conduct_portfolio_analysis(ticker)
                    results.append(
                        f"=== Portfolio News Scan & Debate for {ticker} ===\n"
                        f"{res['synthesis']}\n"
                    )
                    actions_run.append({"type": "debate", "ticker": ticker, "status": "success"})
                except Exception as e:
                    results.append(f"=== Debate for {ticker} Failed ===\nError: {e}\n")
                    actions_run.append({"type": "debate", "ticker": ticker, "status": "error", "error": str(e)})
            else:
                results.append("=== Debate Error ===\nTicker symbol was not specified.\n")
                actions_run.append({"type": "debate", "status": "missing_args"})
                
        elif a_type == "stress_test":
            scenario = a_dict.get("scenario") if isinstance(a_dict, dict) else a_dict.scenario
            if scenario:
                try:
                    report = database.execute_stress_test(scenario)
                    results.append(
                        f"=== Macro Stress Test Report ===\n"
                        f"Scenario: {scenario}\n"
                        f"{report}\n"
                    )
                    actions_run.append({"type": "stress_test", "scenario": scenario, "status": "success"})
                except Exception as e:
                    results.append(f"=== Stress Test Failed ===\nError: {e}\n")
                    actions_run.append({"type": "stress_test", "scenario": scenario, "status": "error", "error": str(e)})
            else:
                results.append("=== Stress Test Error ===\nScenario description was not specified.\n")
                actions_run.append({"type": "stress_test", "status": "missing_args"})
                
        elif a_type == "ingest":
            results.append("=== Ingestion Error ===\nDocument ingestion continuation is not supported from cached actions.\n")
            actions_run.append({"type": "ingest", "status": "unsupported_continuation"})
            
        elif a_type == "performance_analysis":
            try:
                perf_report = database.get_portfolio_performance_summary()
                results.append(perf_report)
                actions_run.append({"type": "performance_analysis", "status": "success"})
            except Exception as e:
                results.append(f"=== Performance Analysis Failed ===\nError: {e}\n")
                actions_run.append({"type": "performance_analysis", "status": "error", "error": str(e)})
                
    results_summary = "\n".join(results)
    if not results_summary:
        print("Executing conversational continuation response...")
        final_response = agent.generate_conversational_response(original_prompt, strategies_str)
        actions_run.append({"type": "conversational"})
    else:
        print("Synthesizing workflow continuation results into chatbot response...")
        final_response = agent.synthesize_chat_response(original_prompt, results_summary, strategies_str)
        
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
    
    return {
        "response": final_response,
        "actions_run": actions_run
    }

