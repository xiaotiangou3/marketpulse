import json
import concurrent.futures
from typing import List, Optional, Callable
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
import config
from .backtest import backtest_strategy_tool
from .trade_tools import execute_paper_trade_tool, create_sandbox_tool

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
        description="Type of action: 'debate', 'stress_test', 'backtest', 'paper_trade', 'performance_analysis', 'ingest', 'add_strategy', 'delete_strategy', 'update_strategy', 'list_strategies', 'create_sandbox', or 'none'"
    )
    ticker: Optional[str] = Field(
        None, 
        description="Ticker symbol (capitalized, e.g. 'MSFT', 'NVDA') if action_type is 'debate', 'backtest', 'paper_trade', or 'ingest'"
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
    qty: Optional[float] = Field(None, description="Number of shares for paper trading (e.g. 10.0)")
    side: Optional[str] = Field("buy", description="Order side for paper trading: 'buy' or 'sell'")
    sandbox_target: Optional[str] = Field(None, description="Optional target sandbox name or ID for paper trading or management")
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
        "Your task is to parse the user's prompt (which may contain slash commands like /debate, /stress, /performance, /backtest, /trade, /help, or freeform natural language) "
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
        "4. 'paper_trade': Triggered by '/trade' command or natural language requesting to place, execute, buy, or sell simulated paper shares (e.g. 'Buy 10 shares of NVDA', 'Sell 5 AAPL in Tech Momentum', 'Yes, place the trade', 'Place a paper trade for 20 TSLA in my RSI sandbox').\n"
        "   - Extract ticker symbol (e.g. NVDA, AAPL). If not explicitly stated in prompt, infer from context or active holdings.\n"
        "   - Extract qty (number of shares). If unspecified, default to 10.0.\n"
        "   - Extract side ('buy' or 'sell', default to 'buy').\n"
        "   - Extract sandbox_target if the prompt mentions a specific sandbox name (e.g. 'RSI sandbox', 'Tech Momentum').\n"
        "5. 'performance_analysis': Triggered by '/performance', '/portfolio', or natural language asking for portfolio valuation, P&L, returns, gains/losses, or performance breakdown.\n"
        "6. 'ingest': PDF transcript uploading (requires a stock ticker symbol. Note: This action should only be triggered if a file has been uploaded, as indicated by has_uploaded_file).\n"
        "7. 'add_strategy': Add a new qualitative investment strategy guideline (requires strategy_text, e.g. 'Limit technology exposure to 40%'). Keep strategy_text concise.\n"
        "8. 'delete_strategy': Delete/remove an existing qualitative strategy guideline (requires strategy_target, which must strictly be the index/number or a short 2-3 word keyword of the rule to delete).\n"
        "9. 'update_strategy': Update/modify an existing strategy guideline (requires strategy_target and strategy_text for the new wording).\n"
        "10. 'list_strategies': Show or list all currently configured strategy guidelines.\n"
        "11. 'create_sandbox': Create a new paper trading sandbox for isolated strategy testing.\n\n"
        "### UPLOADED FILES & PORTFOLIO CONTEXT:\n"
        "- If an uploaded portfolio CSV is attached or present in file_context, evaluate actions (such as performance_analysis or stress_test) against those uploaded CSV holdings.\n"
        "- If it is completely ambiguous whether the user wants to analyze the uploaded file versus the active database portfolio, output 'none' as action_type so the conversational assistant can ask the user for clarification.\n\n"
        "### MULTI-INTENT & HYBRID PROMPTS:\n"
        "- Users can combine slash commands with additional requests in the same prompt (e.g., '/debate MSFT and also run a backtest on RSI').\n"
        "- You must identify ALL requested actions and return them in the sequential order they should execute.\n"
        "- If a prompt is purely conversational with no actions needed, output 'none' as action_type.\n"
        "- For strategy_text, extract the exact wording based on the user's prompt without hallucinating or altering numbers/percentages.\n"
        "- Ensure all action properties are clean, concise, and contain no repetitive looping text."
    )

    
    prompt = f"User Prompt: \"{user_prompt}\"\n"
    if holdings_str:
        prompt += f"\n### CURRENT USER HOLDINGS ###\n{holdings_str}\n"
    if file_context:
        prompt += f"\n### UPLOADED FILE CONTEXT ###\n{file_context}\n"
        
    try:
        response = client.models.generate_content(
            model=config.GENERATIVE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RouterOutput,
                system_instruction=system_instruction,
                temperature=0.0
            )
        )
        return RouterOutput.model_validate_json(response.text)
    except Exception as e:
        print(f"Router parsing error: {e}")
        return RouterOutput(
            explanation="Failed to route specifically, processing conversationally.",
            actions=[ActionItem(action_type="none")]
        )

@llm_retry
def resolve_strategy_match(target_text: str, current_strategies: list) -> Optional[str]:
    """Uses LLM to match a user reference string to an existing strategy ID if exact matching fails."""
    if not current_strategies:
        return None
        
    client = get_gemini_client()
    try:
        prompt = f"User refers to strategy: '{target_text}'.\n\nWhich of the following strategy IDs does this refer to?\n"
        for s in current_strategies:
            prompt += f"- ID: {s['strategy_id']} | Rule: \"{s['strategy_text']}\"\n"
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
        "### UPLOADED FILES & FILE CONTEXT:\n"
        "If a file (such as a CSV or PDF) is attached, its content has been pre-parsed and included in the prompt under '### UPLOADED FILE CONTEXT ###'. "
        "You DO have direct access to its contents. NEVER tell the user that you cannot read, parse, or access external/local files, PDFs, or CSVs. "
        "Instead, read the pre-parsed text in '### UPLOADED FILE CONTEXT ###' and answer the user's question about the file directly.\n\n"
        "### QUANTITATIVE STRATEGY INTERPRETATION & BACKTESTING:\n"
        "When users suggest strategic rules, trading conditions, or ask to validate a strategy idea on an asset, interpret their intent and invoke the `backtest_strategy_tool`:\n"
        "- **Moving Averages (SMA/EMA)**: e.g., 'Should I sell NVDA if it drops below 20-day SMA?' -> `strategy_type='sma_cross'`, `short_window=20`, `long_window=50` (or `strategy_type='ema_cross'`).\n"
        "- **RSI (Relative Strength Index)**: e.g., 'Buy NVDA when RSI < 30 and sell when RSI > 70' -> `strategy_type='rsi'`, `rsi_period=14`, `rsi_oversold=30`, `rsi_overbought=70`.\n"
        "- **MACD Crossover**: e.g., 'Test MACD crossover on AAPL' -> `strategy_type='macd'`, `macd_fast=12`, `macd_slow=26`, `macd_signal=9`.\n"
        "- **Bollinger Bands**: e.g., 'Buy TSLA when price hits lower Bollinger band' -> `strategy_type='bollinger'`, `bb_window=20`, `bb_std=2.0`.\n"
        "- **Price Breakouts**: e.g., 'Buy MSFT on 20-day high breakout' -> `strategy_type='breakout'`, `breakout_window=20`.\n\n"
        "### PROACTIVE MULTI-SANDBOX PAPER TRADING:\n"
        "Whenever a strategy backtest shows a positive return or outperformance over the benchmark, proactively ask the user if they would like to execute a paper trade for that stock with a default of 10 shares into the bound strategy sandbox (e.g., 'Would you like me to place a paper buy order for 10 shares of NVDA in the RSI Strategy Sandbox?').\n\n"
        "### DIRECT PAPER TRADING EXECUTION & SANDBOX MANAGEMENT:\n"
        "1. If the user confirms or gives a direct trade command (e.g., 'Buy 10 shares of NVDA', 'Yes, place the trade', 'Sell 5 AAPL', 'Execute paper trade'), invoke `execute_paper_trade_tool` immediately.\n"
        "2. If no sandboxes exist when requesting a trade, offer to create a new sandbox for this strategy (or invoke `create_sandbox_tool`).\n"
        "3. If the user asks to create a strategy sandbox, invoke `create_sandbox_tool` (maximum 10 sandboxes allowed).\n\n"
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
        
    return generate_ai_response(prompt, system_instruction, tools=[backtest_strategy_tool, execute_paper_trade_tool, create_sandbox_tool])

@llm_retry
def synthesize_chat_response(user_prompt: str, results_summary: str, strategies_str: str = "", file_context: str = None, status_callback: Optional[Callable[[str, Optional[str]], None]] = None) -> str:
    if status_callback:
        status_callback("📝 Synthesizing investment insights...", "Synthesizing multi-agent outputs and testing strategy constraints...")
    system_instruction = (
        "You are a friendly, highly professional AI Investment Assistant named MarketPulse AI. "
        "You have executed tools (debates, macro stress tests, multi-indicator backtests, paper trades, sandbox creations, document ingestion, or strategy modifications) to satisfy the user's request. "
        "Your goal is to synthesize the outcomes of these tools into a clean, action-oriented, and "
        "insightful conversational response. When backtests have been conducted, weave the key metrics (returns vs. benchmark, win rate, drawdown, rule definitions) "
        "into your qualitative commentary. "
        "Whenever a strategy backtest shows a positive return or outperformance over the benchmark, proactively ask the user if they would like to execute a paper buy order for 10 shares of that stock into the corresponding strategy sandbox. "
        "If a paper trade was executed, clearly confirm the target sandbox name and order execution details in your response.\n\n"
        "### UPLOADED FILES & FILE CONTEXT:\n"
        "If a file (such as a CSV or PDF) is attached, its content has been pre-parsed and included in the prompt under '### UPLOADED FILE CONTEXT ###'. "
        "You DO have direct access to its contents. NEVER tell the user that you cannot read, parse, or access external/local files, PDFs, or CSVs. "
        "Instead, read the pre-parsed text in '### UPLOADED FILE CONTEXT ###' and answer the user's question about the file directly.\n\n"
        "Always check these outcomes against the user's qualitative strategy rules "
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
        
    return generate_ai_response(prompt, system_instruction, tools=[backtest_strategy_tool, execute_paper_trade_tool, create_sandbox_tool])
