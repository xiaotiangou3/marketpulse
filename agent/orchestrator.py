import json
import concurrent.futures
from typing import List, Optional, Callable
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
import config
from .backtest import backtest_strategy_tool

_client = None

llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)

def get_gemini_client():
    global _client
    if _client is None:
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY must be set in environment variables.")
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client

@llm_retry
def generate_ai_response(prompt: str, system_instruction: str = None, tools: list = None) -> str:
    client = get_gemini_client()
    
    # We construct config for the request if a system instruction or tools are provided
    req_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.4,
        tools=tools
    )
        
    response = client.models.generate_content(
        model=config.GENERATIVE_MODEL,
        contents=prompt,
        config=req_config
    )
    return response.text

class ActionItem(BaseModel):
    action_type: str = Field(
        description="Must be one of: 'debate', 'stress_test', 'backtest', 'ingest', 'performance_analysis', 'add_strategy', 'delete_strategy', 'update_strategy', 'list_strategies', or 'none'"
    )
    ticker: Optional[str] = Field(
        None, 
        description="Ticker symbol (capitalized, e.g. 'MSFT', 'NVDA') if action_type is 'debate', 'backtest', or 'ingest'"
    )
    scenario: Optional[str] = Field(
        None, 
        description="Scenario description if action_type is 'stress_test'"
    )
    strategy_type: Optional[str] = Field(
        None,
        description="For 'backtest' action: one of 'rsi', 'macd', 'ema_cross', 'sma_cross', 'bollinger', 'breakout'. Defaults to 'sma_cross' if simple moving average."
    )
    timeframe: Optional[str] = Field("1y", description="Timeframe for backtest: '6mo', '1y', '2y', '5y'")
    short_window: Optional[int] = Field(20, description="Short lookback window for SMA/EMA")
    long_window: Optional[int] = Field(50, description="Long lookback window for SMA/EMA")
    rsi_period: Optional[int] = Field(14, description="RSI period")
    rsi_oversold: Optional[float] = Field(30.0, description="RSI oversold entry threshold (e.g. 30.0)")
    rsi_overbought: Optional[float] = Field(70.0, description="RSI overbought exit threshold (e.g. 70.0)")
    macd_fast: Optional[int] = Field(12, description="MACD fast period")
    macd_slow: Optional[int] = Field(26, description="MACD slow period")
    macd_signal: Optional[int] = Field(9, description="MACD signal line period")
    bb_window: Optional[int] = Field(20, description="Bollinger bands period")
    bb_std: Optional[float] = Field(2.0, description="Bollinger bands standard deviation")
    breakout_window: Optional[int] = Field(20, description="Breakout lookback window")
    strategy_text: Optional[str] = Field(
        None,
        description="New or updated strategy text if action_type is 'add_strategy' or 'update_strategy'. Keep it concise and avoid repeating phrases."
    )
    strategy_target: Optional[str] = Field(
        None,
        description="Keep it extremely short: strictly either a rule number (e.g. '1') or a short 2-3 word keyword (e.g. 'tech limit'). Do not write long explanations or repeat phrases."
    )

class RouterOutput(BaseModel):
    explanation: str = Field(
        description="Brief sentence explaining what the user wants and how we will route it. Keep it concise."
    )
    actions: List[ActionItem] = Field(
        description="The sequential list of actions to execute."
    )

@llm_retry
def route_user_intent(user_prompt: str, has_uploaded_file: bool, file_context: str = None, holdings_str: str = None, status_callback: Optional[Callable[[str, Optional[str]], None]] = None) -> RouterOutput:
    if status_callback:
        status_callback("🧠 Parsing prompt & analyzing intent...", "Evaluating query against portfolio context, rules, and slash commands...")
    client = get_gemini_client()
    
    system_instruction = (
        "You are an AI Routing Assistant for the MarketPulse AI financial system. "
        "Your task is to parse the user's prompt (which may contain slash commands like /debate, /stress, /performance, /backtest, /help, or freeform natural language) "
        "and decompose it into a structured sequence of actions to execute.\n\n"
        "### SLASH COMMANDS & ACTIONS:\n"
        "1. 'debate': Triggered by '/debate <description>' or natural language requesting a Bull vs. Bear debate, stock analysis, or news catalysts.\n"
        "   - Extract the stock ticker symbol mentioned or implied (e.g. MSFT, AAPL, NVDA). Convert to uppercase ticker symbol.\n"
        "   - If no specific ticker is mentioned in the prompt, pick the primary ticker from the user's active holdings context.\n"
        "2. 'stress_test': Triggered by '/stress <description>', '/stress_test <description>', or natural language asking about macro scenarios, interest rate hikes, inflation shocks, recessions, oil price surges, etc.\n"
        "   - Extract the full scenario text into the 'scenario' field (e.g. 'Federal Reserve hikes interest rates 50bps and oil surges').\n"
        "3. 'backtest': Triggered by '/backtest <params>' or natural language asking to test/validate a technical strategy, trading rule, or indicator condition (e.g., 'Is buying NVDA when RSI drops below 30 and sell when it crosses above 70 a good strategy?').\n"
        "   - Extract the ticker symbol (e.g. NVDA, AAPL, MSFT).\n"
        "   - Identify 'strategy_type': 'rsi' for RSI conditions, 'macd' for MACD crossover, 'ema_cross' for exponential moving averages, 'sma_cross' for simple moving averages, 'bollinger' for Bollinger bands, 'breakout' for price breakouts.\n"
        "   - Extract indicator parameters (e.g. rsi_period=14, rsi_oversold=30, rsi_overbought=70, short_window, long_window, macd_fast, macd_slow, macd_signal, bb_window, bb_std, breakout_window, timeframe).\n"
        "   - If the user's trading idea is completely vague or has no specific technical indicator, output 'none' so the conversational assistant can clarify.\n"
        "4. 'performance_analysis': Triggered by '/performance', '/portfolio', or natural language asking for portfolio valuation, P&L, returns, gains/losses, or performance breakdown.\n"
        "5. 'ingest': PDF transcript uploading (requires a stock ticker symbol. Note: This action should only be triggered if a file has been uploaded, as indicated by has_uploaded_file).\n"
        "6. 'add_strategy': Add a new qualitative investment strategy guideline (requires strategy_text, e.g. 'Limit technology exposure to 40%'). Keep strategy_text concise.\n"
        "7. 'delete_strategy': Delete/remove an existing qualitative strategy guideline (requires strategy_target, which must strictly be the index/number or a short 2-3 word keyword of the rule to delete).\n"
        "8. 'update_strategy': Update/modify an existing strategy guideline (requires strategy_target and strategy_text for the new wording).\n"
        "9. 'list_strategies': Show or list all currently configured strategy guidelines.\n\n"
        "### MULTI-INTENT & HYBRID PROMPTS:\n"
        "- Users can combine slash commands with additional requests in the same prompt (e.g., '/debate MSFT and also run a backtest on RSI').\n"
        "- You must identify ALL requested actions and return them in the sequential order they should execute.\n"
        "- If a prompt is purely conversational with no actions needed, output 'none' as action_type.\n"
        "- For strategy_text, extract the exact wording based on the user's prompt without hallucinating or altering numbers/percentages.\n"
        "- Ensure all action properties are clean, concise, and contain no repetitive looping text."
    )
    
    prompt = (
        f"User Prompt: \"{user_prompt}\"\n"
        f"Has Uploaded File Context: {has_uploaded_file}\n"
    )
    if holdings_str:
        prompt += f"### USER CURRENT PORTFOLIO HOLDINGS ###\n{holdings_str}\n"
    if file_context:
        prompt += f"### UPLOADED FILE CONTEXT ###\n{file_context}\n"
    
    response = client.models.generate_content(
        model=config.GENERATIVE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=RouterOutput
        )
    )
    
    try:
        output = RouterOutput.model_validate_json(response.text)
        if status_callback:
            action_names = [a.action_type for a in output.actions if a.action_type.lower() != "none"]
            summary = f"Identified {len(action_names)} action(s): {', '.join(action_names)}" if action_names else "Identified conversational query"
            status_callback("🎯 Plan determined", summary)
        return output
    except Exception as e:
        print(f"Error parsing router JSON output: {response.text}, error: {e}")
        return RouterOutput(explanation="Fallback routing due to parse error.", actions=[])

def resolve_strategy_match(target: str, current_strategies: List[dict]) -> Optional[str]:
    """
    Identifies which strategy from the list matches the user's reference string (target).
    Checks numeric index first, then text substring, and falls back to Gemini API semantic match.
    """
    if not current_strategies:
        return None
        
    # 1. Try numeric index matching first
    clean_target = target.strip().replace("#", "")
    if clean_target.isdigit():
        idx = int(clean_target) - 1
        if 0 <= idx < len(current_strategies):
            return current_strategies[idx]['strategy_id']
            
    # 2. Try simple exact/substring case-insensitive match on text
    for s in current_strategies:
        if target.lower() in s['strategy_text'].lower() or s['strategy_text'].lower() in target.lower():
            return s['strategy_id']
            
    # 3. Use Gemini to resolve semantically
    try:
        client = get_gemini_client()
        prompt = f"We need to find which strategy in the list matches the user's reference: \"{target}\".\n\nList of strategies:\n"
        for idx, s in enumerate(current_strategies):
            prompt += f"- Index: {idx+1}, ID: {s['strategy_id']}, Text: \"{s['strategy_text']}\"\n"
            
        prompt += "\nRespond ONLY with the matching strategy's ID from the list. If none match, respond with 'none'. Do not include any other text."
        
        response = client.models.generate_content(
            model=config.GENERATIVE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        res_text = response.text.strip().lower()
        
        for s in current_strategies:
            if str(s['strategy_id']).lower() in res_text:
                return s['strategy_id']
    except Exception as e:
        print(f"Semantic match fallback error: {e}")
        
    return None

@llm_retry
def generate_conversational_response(user_prompt: str, strategies_str: str, file_context: str = None, status_callback: Optional[Callable[[str, Optional[str]], None]] = None) -> str:
    if status_callback:
        status_callback("💬 Formulating quantitative & conversational response...", "Analyzing strategy constraints and verifying historical data...")
    system_instruction = (
        "You are a quantitative portfolio assistant named MarketPulse AI.\n\n"
        "### QUANTITATIVE STRATEGY INTERPRETATION & BACKTESTING:\n"
        "When users suggest strategic rules, trading conditions, or ask to validate a strategy idea on an asset, interpret their intent and invoke the `backtest_strategy_tool`:\n"
        "- **Moving Averages (SMA/EMA)**: e.g., 'Should I sell NVDA if it drops below 20-day SMA?' -> `strategy_type='sma_cross'`, `short_window=20`, `long_window=50` (or `strategy_type='ema_cross'`).\n"
        "- **RSI (Relative Strength Index)**: e.g., 'Buy NVDA when RSI < 30 and sell when RSI > 70' -> `strategy_type='rsi'`, `rsi_period=14`, `rsi_oversold=30`, `rsi_overbought=70`.\n"
        "- **MACD Crossover**: e.g., 'Test MACD crossover on AAPL' -> `strategy_type='macd'`, `macd_fast=12`, `macd_slow=26`, `macd_signal=9`.\n"
        "- **Bollinger Bands**: e.g., 'Buy TSLA when price hits lower Bollinger band' -> `strategy_type='bollinger'`, `bb_window=20`, `bb_std=2.0`.\n"
        "- **Price Breakouts**: e.g., 'Buy MSFT on 20-day high breakout' -> `strategy_type='breakout'`, `breakout_window=20`.\n\n"
        "### IMPORTANT: AMBIGUITY HANDLING:\n"
        "If you are unsure about what strategy the user is talking about, or if the user's trading idea is too vague to determine a specific technical rule or indicator, simply tell the user clearly that you are unsure and ask them to specify the strategy or indicator parameters they wish to test. Do not invent a random strategy or make unnecessary tool calls when unsure.\n\n"
        "### QUALITATIVE STRATEGY RULES:\n"
        "Always evaluate and adhere to the active qualitative strategy rules listed below when answering user queries. "
        "If the query involves investing, portfolio holdings, or market events, ensure your advice "
        "checks and respects these rules, flagging potential compliance conflicts if any exist.\n\n"
        "Note: If the user explicitly asks you to add, delete, list, or update strategy rules, prioritize the user's instructions. "
        "Do not refuse, block, or advise against modifying the strategies, as the user is the ultimate authority who manages them.\n\n"
        f"### ACTIVE QUALITATIVE STRATEGY RULES:\n{strategies_str}\n\n"
        "Keep your reply friendly, professional, conversational, and educational. "
        "Remind the user that this is for educational simulation only and is not official financial advice."
    )
    
    prompt = user_prompt
    if file_context:
        prompt += f"\n\n### UPLOADED FILE CONTEXT ###\n{file_context}\n"
        
    return generate_ai_response(prompt, system_instruction, tools=[backtest_strategy_tool])

@llm_retry
def synthesize_chat_response(user_prompt: str, results_summary: str, strategies_str: str = "", file_context: str = None, status_callback: Optional[Callable[[str, Optional[str]], None]] = None) -> str:
    if status_callback:
        status_callback("📝 Synthesizing investment insights...", "Synthesizing multi-agent outputs and testing strategy constraints...")
    system_instruction = (
        "You are a friendly, highly professional AI Investment Assistant named MarketPulse AI. "
        "You have executed tools (debates, macro stress tests, multi-indicator backtests, document ingestion, or strategy modifications) to satisfy the user's request. "
        "Your goal is to synthesize the outcomes of these tools into a clean, action-oriented, and "
        "insightful conversational response. When backtests have been conducted, weave the key metrics (returns vs. benchmark, win rate, drawdown, rule definitions) "
        "into your qualitative commentary. Always check these outcomes against the user's qualitative strategy rules "
        "listed below, and emphasize how they align or conflict.\n\n"
        "If you are unsure about what strategy the user intended, communicate that clearly without guessing or taking unnecessary speculative actions.\n\n"
        "Note: If the user explicitly requested strategy additions, updates, or deletions, confirm the successful drafting of these strategy changes "
        "without advising against or blocking them, as the user is the ultimate authority who manages the strategies.\n\n"
        f"### ACTIVE QUALITATIVE STRATEGY RULES:\n{strategies_str}\n\n"
        "Weave the insights together naturally. Be helpful, and emphasize that this is for educational simulation only and is not official financial advice."
    )
    
    prompt = (
        f"User Prompt: \"{user_prompt}\"\n\n"
        f"TOOL EXECUTION RESULTS SUMMARY:\n"
        f"{results_summary}\n"
    )
    
    if file_context:
        prompt += f"\n### UPLOADED FILE CONTEXT ###\n{file_context}\n"
        
    return generate_ai_response(prompt, system_instruction, tools=[backtest_strategy_tool])



