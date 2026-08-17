import json
import concurrent.futures
from typing import List, Optional, Callable, Union
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
def generate_ai_response(prompt: Union[str, list], system_instruction: str = None, tools: list = None) -> str:
    client = get_gemini_client()
    
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
        description="Type of action: 'debate', 'stress_test', 'backtest', 'paper_trade', 'ingest', 'list_strategies', 'create_sandbox', or 'none'"
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
    csv_ticker_col: Optional[str] = Field(
        None,
        description="The custom column name for stock ticker/symbol in the CSV file, if the user explicitly clarifies/provides it (e.g. 'Symbol', 'Ticker name')."
    )
    csv_shares_col: Optional[str] = Field(
        None,
        description="The custom column name for shares/quantity in the CSV file, if the user explicitly clarifies/provides it (e.g. 'Quantity', 'Qty', 'Shares count')."
    )
    csv_cost_col: Optional[str] = Field(
        None,
        description="The custom column name for cost basis/purchase price in the CSV file, if the user explicitly clarifies/provides it (e.g. 'Avg Cost', 'Purchase Price')."
    )

class RequestContract(BaseModel):
    primary_goal: str = Field(description="One sentence describing the user's actual requested outcome.")
    requested_tasks: List[str] = Field(default_factory=list, description="Concrete tasks explicitly requested by the user.")
    explicitly_prohibited: List[str] = Field(default_factory=list, description="Actions, assumptions, or additions the user explicitly rejected or did not ask for.")
    source_required: bool = Field(False, description="True when the answer must be grounded primarily in uploaded/provided source material.")
    source_only: bool = Field(False, description="True when the user asks about a provided source and no outside interpretation should replace or override it.")
    needs_tool: bool = Field(False, description="True only when a tool is required by the user's explicit request.")
    allowed_actions: List[str] = Field(default_factory=list, description="Only actions explicitly requested or strictly necessary to satisfy the request.")
    answer_scope: str = Field(description="Scope restriction for the final answer.")

class RouterOutput(BaseModel):
    explanation: str = Field(
        description="Brief sentence explaining what the user wants and how we will route it. Keep it concise."
    )
    actions: List[ActionItem] = Field(
        description="The sequential list of actions to execute."
    )
    contract: RequestContract = Field(
        default_factory=lambda: RequestContract(
            primary_goal="Process the request.",
            requested_tasks=[],
            explicitly_prohibited=[],
            source_required=False,
            source_only=False,
            needs_tool=False,
            allowed_actions=[],
            answer_scope="general"
        ),
        description="The non-negotiable user-intent contract extracted from the prompt."
    )

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="True if the response has zero violations of instruction compliance, source grounding, or parameter preservation. False if any violation is present.")
    violations: List[str] = Field(default_factory=list, description="List of specific violations found, if any.")
    corrected_response: str = Field(description="If is_valid is False, this MUST be the corrected response which removes all violations and is completely faithful. If is_valid is True, this should be the original response exactly as-is.")

def build_behavior_contract(user_prompt: str, file_context: str = None, strategies_str: str = "", contract: Optional[RequestContract] = None) -> str:
    """Shared behavioral contract used by the conversational and synthesis layers."""
    base_instructions = (
        "### NON-NEGOTIABLE USER-INTENT CONTRACT ###\n"
        "The user's request is the source of truth for what task you perform. Do not replace the user's task with a task you personally consider more useful. "
        "Do not broaden, reinterpret, or redirect the request unless required for correctness or safety.\n\n"
        "PRIORITY ORDER: (1) system/developer instructions, (2) explicit user instructions, (3) provided source material, (4) active strategy rules, (5) general model knowledge.\n"
        "When sources are provided, source material is evidence, not an instruction to invent missing content. Preserve the source's terminology, structure, and level of specificity.\n\n"
        "### SOURCE-GROUNDING RULES ###\n"
        "1. If the user asks to analyze/review/evaluate a supplied document, answer that document first and directly.\n"
        "2. Never attribute facts, rules, allocations, risk profiles, strategies, or requirements to a source unless the source actually states them or they are an explicitly labeled inference.\n"
        "3. Never fill gaps in a source with a typical industry convention. If the source does not specify something, say: 'The document does not specify this.'\n"
        "4. Distinguish clearly between EXPLICIT (stated in source), INFERENCE (logical interpretation), and EXTERNAL CONTEXT (general knowledge).\n"
        "5. Do not silently correct, reconcile, replace, or rewrite the user's source material.\n\n"
        "### NO-UNSOLICITED-ACTION RULES ###\n"
        "6. Do not run a backtest unless the user explicitly asks for a backtest, quantitative validation, historical test, or equivalent.\n"
        "7. Do not place, suggest, or prepare a paper trade merely because a backtest was positive unless the user explicitly asks to trade or explicitly invites that next step.\n"
        "8. Do not create or modify a sandbox unless the user explicitly requests it.\n"
        "9. Do not add unrelated analysis just because it is available. A useful extra point is allowed only when it is directly necessary to answer the user's question.\n"
        "10. If the user asks whether something is useful, evaluate usefulness; do not automatically execute or backtest it.\n\n"
        "### EVIDENCE & CONCLUSION RULES ###\n"
        "11. Never generalize from a single security, short period, or single backtest beyond what that evidence supports.\n"
        "12. Separate 'what the evidence shows' from 'what it does not show'.\n"
        "13. Do not frame two approaches as competitors unless the user asks for a comparison or the approaches truly conflict.\n"
        "14. If a quantitative strategy and an IPS can coexist, say so rather than inventing a false either/or.\n\n"
    )
    
    if contract:
        contract_info = (
            "### STRICT CONTRACT LIMITS FROM ROUTER ###\n"
            f"- Primary Goal: {contract.primary_goal}\n"
            f"- Explicitly Requested: {', '.join(contract.requested_tasks) if contract.requested_tasks else 'None'}\n"
            f"- Explicitly Prohibited: {', '.join(contract.explicitly_prohibited) if contract.explicitly_prohibited else 'None'}\n"
            f"- Source Required: {contract.source_required}\n"
            f"- Source Only: {contract.source_only}\n"
            f"- Scope: {contract.answer_scope}\n\n"
        )
        base_instructions = contract_info + base_instructions

    return (
        base_instructions +
        f"### USER REQUEST ###\n{user_prompt}\n\n"
        + (f"### PROVIDED SOURCE MATERIAL ###\n{file_context}\n\n" if file_context else "")
        + (f"### ACTIVE STRATEGY RULES ###\n{strategies_str}\n" if strategies_str else "")
    )

@llm_retry
def validate_response(
    raw_response: str,
    user_prompt: str,
    contract: RequestContract,
    file_context: Optional[str] = None
) -> str:
    """
    Validates a generated response against the RequestContract, user prompt, and file context.
    If violations are detected, it returns a corrected response.
    """
    client = get_gemini_client()
    
    validation_instruction = (
        "You are the strict Output Validator for MarketPulse AI.\n"
        "Your task is to inspect the generated raw response and verify if it complies with the user intent contract.\n\n"
        "### VALIDATION CHECKLIST:\n"
        "1. Instruction Compliance:\n"
        "   - Did the response answer the actual requested question?\n"
        "   - Did it add an unrelated task, suggest unsolicited paper trades, or recommend things the user did not request?\n"
        "2. Source Grounding:\n"
        "   - Did it claim something came from the uploaded file when it didn't?\n"
        "   - Did it contradict the source file?\n"
        "   - Did it introduce unsupported industry conventions or assumptions to fill gaps in the file?\n"
        "3. Parameter Preservation:\n"
        "   - Were any percentages, thresholds, tickers, quantities, or timeframes from the user prompt changed?\n\n"
        "### VIOLATION HANDLING:\n"
        "If you find ANY violation:\n"
        "- Set is_valid to False.\n"
        "- List the specific violations.\n"
        "- In corrected_response, provide a rewritten version of the response that removes all violations, retains the original user constraints, and strictly adheres to the grounding rules.\n"
        "If the response is fully valid and has no violations:\n"
        "- Set is_valid to True.\n"
        "- Set corrected_response to the original response EXACTLY as-is (do not change a single character)."
    )
    
    prompt = (
        f"### USER PROMPT ###\n{user_prompt}\n\n"
        f"### REQUEST CONTRACT ###\n"
        f"- Primary Goal: {contract.primary_goal}\n"
        f"- Explicitly Requested: {', '.join(contract.requested_tasks) if contract.requested_tasks else 'None'}\n"
        f"- Explicitly Prohibited/Not Requested: {', '.join(contract.explicitly_prohibited) if contract.explicitly_prohibited else 'None'}\n"
        f"- Source Required: {contract.source_required}\n"
        f"- Source Only: {contract.source_only}\n"
        f"- Allowed Actions: {', '.join(contract.allowed_actions) if contract.allowed_actions else 'None'}\n\n"
    )
    if file_context:
        prompt += f"### UPLOADED SOURCE MATERIAL ###\n{file_context}\n\n"
    
    prompt += (
        f"### GENERATED RESPONSE TO VALIDATE ###\n"
        f"{raw_response}\n"
    )
    
    try:
        response = client.models.generate_content(
            model=config.GENERATIVE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ValidationResult,
                system_instruction=validation_instruction,
                temperature=0.0
            )
        )
        res = ValidationResult.model_validate_json(response.text)
        if not res.is_valid:
            print(f"Validator detected violations: {res.violations}")
            return res.corrected_response
        return raw_response
    except Exception as e:
        print(f"Validation error: {e}")
        return raw_response

@llm_retry
def route_user_intent(
    user_prompt: str,
    has_uploaded_file: bool,
    file_context: str = None,
    holdings_str: str = None,
    status_callback: Optional[Callable[[str, Optional[str]], None]] = None,
    chat_history: Optional[List[dict]] = None
) -> RouterOutput:
    if status_callback:
        status_callback("🧠 Parsing prompt & analyzing intent...", "Extracting only the actions explicitly requested by the user...")
    client = get_gemini_client()

    system_instruction = (
        "You are the deterministic intent router for MarketPulse AI. Your job is NOT to solve the user's question. "
        "Your job is to identify exactly what the user asked the application to do and return only the minimum required actions, and generate a strict behavioral contract.\n\n"
        "### HARD ROUTING RULES ###\n"
        "1. Treat explicit user instructions as authoritative. Do not invent additional tasks.\n"
        "2. Do not convert an analysis/review/explanation request into a backtest, trade, debate, stress test, or sandbox action unless explicitly requested.\n"
        "3. If the user asks to analyze an uploaded document/PDF/IPS, route to 'none' unless a separate concrete tool action is explicitly requested.\n"
        "4. 'none' means conversational processing is required; it does NOT mean the request is invalid.\n"
        "5. For multi-intent prompts, include every explicitly requested action in the exact order the user requests them. Do not add actions before or after them.\n"
        "6. Preserve the user's numbers, thresholds, tickers, wording, and constraints exactly.\n"
        "7. Never infer a ticker from holdings unless the user's request clearly refers to that holding.\n"
        "8. Never default to a backtest just because the user mentions a strategy. Mentioning or discussing a strategy is not the same as asking to test it.\n"
        "9. Never default to paper trading. A positive backtest is not a request to trade.\n"
        "10. When information is missing, leave the field empty rather than inventing a value, except for defaults explicitly defined by the application contract.\n"
        "11. Do not route to performance_analysis, add_strategy, delete_strategy, or update_strategy. These action types are no longer supported. Handle any requests to check performance or manage strategies conversationally (route to 'none') or refer the user to the UI buttons. You can route to 'list_strategies' when the user explicitly requests to list/view strategy rules.\n\n"
        "### ACTION DEFINITIONS ###\n"
        "- debate: only an explicit request for bull/bear debate, stock debate, or equivalent structured debate.\n"
        "- stress_test: only an explicit request to test a macro/scenario shock.\n"
        "- backtest: only an explicit request to test/validate a quantitative trading rule on historical data.\n"
        "- paper_trade: only an explicit request to buy/sell/execute simulated shares.\n"
        "- ingest: only when the application workflow requires document ingestion (either a file is actually uploaded, or the user is clarifying the custom column mapping/format of a previously uploaded CSV file).\n"
        "- list_strategies: only when explicitly requested to list, view, or display qualitative investment strategy rules/guidelines.\n"
        "- create_sandbox: only when explicitly requested to create a sandbox.\n"
        "- none: conversational question, source analysis, explanation, comparison, interpretation, check performance, or any request needing no tool action.\n\n"
        "### UPLOADED FILE RULE ###\n"
        "If the user asks to analyze, critique, explain, or evaluate an uploaded qualitative document such as an IPS, do NOT route to backtest. "
        "The conversational layer must answer from the supplied file contents.\n\n"
        "### CUSTOM CSV COLUMN CLARIFICATION RULE ###\n"
        "If the user is clarifying the column names or format of a CSV file (e.g. 'Ticker = Symbol', 'format is Symbol, Qty, Cost Basis', 'Avg Price = Cost', etc.), you MUST route to 'ingest' action. Extract the custom column names into csv_ticker_col, csv_shares_col, and csv_cost_col. This is part of the ingestion/overwrite workflow, not a conversational message or a backtest. Do not route to backtest unless explicitly requested.\n\n"
        "### MULTI-INTENT RULE ###\n"
        "Identify ALL requested actions, but do not add helpful actions the user did not request.\n\n"
        "### CONTRACT GENERATION RULES ###\n"
        "You must generate a RequestContract that governs how the conversational and synthesis layers will answer the request.\n"
        "1. primary_goal: A clear, single-sentence summary of what the user wants to achieve.\n"
        "2. requested_tasks: The list of tasks the user explicitly requested.\n"
        "3. explicitly_prohibited: Identify any actions, assumptions, or additions the user explicitly rejected or did not ask for (e.g., if the user did not ask to trade, 'recommend paper trade' or 'execute trade' is prohibited; if the user asked a qualitative question, 'run quantitative backtest' is prohibited; if the user asked for analysis only, 'execute/recommend trade' is prohibited).\n"
        "4. source_required: True if the user's query asks about an uploaded document/file, the portfolio context, or active strategies.\n"
        "5. source_only: True if the user asks specifically about a provided source and no outside general knowledge should override it.\n"
        "6. needs_tool: True only when a tool is required by the user's explicit request.\n"
        "7. allowed_actions: The list of action types explicitly requested (e.g. ['debate', 'stress_test', 'backtest', 'paper_trade']). This MUST match the action_types of the actions list.\n"
        "8. answer_scope: A brief description of the scope of the final answer."
    )

    contents = []
    if chat_history:
        for msg in chat_history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
            
    current_prompt_with_context = f"User Prompt:\n{user_prompt}\n"
    if holdings_str:
        current_prompt_with_context += f"\n### CURRENT USER HOLDINGS ###\n{holdings_str}\n"
    if file_context:
        current_prompt_with_context += f"\n### UPLOADED FILE CONTEXT ###\n{file_context}\n"
        
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=current_prompt_with_context)]))

    try:
        response = client.models.generate_content(
            model=config.GENERATIVE_MODEL,
            contents=contents,
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
            explanation="Processing conversationally because no reliable tool action could be extracted.",
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
def generate_conversational_response(
    user_prompt: str,
    strategies_str: str,
    file_context: str = None,
    status_callback: Optional[Callable[[str, Optional[str]], None]] = None,
    allowed_actions: Optional[List[str]] = None,
    contract: Optional[RequestContract] = None,
    chat_history: Optional[List[dict]] = None,
) -> str:
    if status_callback:
        status_callback("💬 Formulating response...", "Following the user's requested scope and grounding claims in the supplied context...")

    allowed_actions = allowed_actions or []
    behavior_contract = build_behavior_contract(user_prompt, file_context, strategies_str, contract)
    allowed = ", ".join(allowed_actions) if allowed_actions else "NONE"

    system_instruction = (
        "You are MarketPulse AI, a quantitative portfolio assistant.\n\n"
        + behavior_contract + "\n"
        "### TOOL ACCESS CONTROL ###\n"
        f"Allowed action types for this turn: {allowed}\n"
        "You MUST NOT call any tool whose action is not in the allowed list. If the allowed list is NONE, do not call any tool.\n"
        "A tool being technically available does not mean you are authorized to use it.\n\n"
        "### ANSWER METHOD ###\n"
        "First answer the user's actual question. Only include analysis necessary to satisfy that request.\n"
        "If a supplied IPS/document is being evaluated, quote/paraphrase its actual rules accurately and explain their usefulness one by one.\n"
        "Use labels such as 'The document states', 'This implies', and 'The document does not specify' when needed to prevent source/inference confusion.\n"
        "Never invent an asset allocation, risk profile, benchmark, strategy, or restriction that is absent from the source.\n"
        "If the user asks whether strategies are useful, discuss benefits, limitations, and fit with the stated objective; do not automatically run a backtest or recommend a trade.\n\n"
        "### QUANTITATIVE EVIDENCE RULES ###\n"
        "If and only if an allowed backtest action exists, interpret results conservatively. A backtest result is evidence for the tested configuration and period, not proof of universal superiority.\n"
        "Do not claim an IPS, trading strategy, or indicator is generally superior based on one asset or one period.\n\n"
        "Be concise, accurate, and faithful to the user's wording and requested scope."
    )

    tools = []
    if "backtest" in allowed_actions:
        tools.append(backtest_strategy_tool)
    if "paper_trade" in allowed_actions:
        tools.append(execute_paper_trade_tool)
    if "create_sandbox" in allowed_actions:
        tools.append(create_sandbox_tool)

    contents = []
    if chat_history:
        for msg in chat_history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)]))

    raw_response = generate_ai_response(
        contents,
        system_instruction,
        tools=tools or None,
    )
    
    # If no contract is provided, construct a basic default contract for validation
    validation_contract = contract or RequestContract(
        primary_goal=user_prompt[:100],
        requested_tasks=[],
        explicitly_prohibited=["suggest trade", "execute trade", "recommend trade", "run backtest"],
        source_required=bool(file_context),
        source_only=False,
        needs_tool=len(tools) > 0,
        allowed_actions=allowed_actions,
        answer_scope="conversational"
    )
    
    return validate_response(
        raw_response=raw_response,
        user_prompt=user_prompt,
        contract=validation_contract,
        file_context=file_context
    )

@llm_retry
def synthesize_chat_response(
    user_prompt: str,
    results_summary: str,
    strategies_str: str = "",
    file_context: str = None,
    status_callback: Optional[Callable[[str, Optional[str]], None]] = None,
    allowed_actions: Optional[List[str]] = None,
    contract: Optional[RequestContract] = None,
    chat_history: Optional[List[dict]] = None,
) -> str:
    if status_callback:
        status_callback("📝 Synthesizing response...", "Preserving the requested scope and checking conclusions against the evidence...")

    allowed_actions = allowed_actions or []
    behavior_contract = build_behavior_contract(user_prompt, file_context, strategies_str, contract)
    allowed = ", ".join(allowed_actions) if allowed_actions else "NONE"

    system_instruction = (
        "You are the final-response synthesis layer for MarketPulse AI.\n\n"
        + behavior_contract + "\n"
        "### SYNTHESIS RULES ###\n"
        "1. The TOOL EXECUTION RESULTS are evidence, not permission to invent new tasks.\n"
        "2. Answer only the user's requested question/request.\n"
        "3. Do not introduce a new trade, backtest, sandbox, debate, or stress test unless that action was explicitly requested and is present in the allowed action list.\n"
        "4. Do not turn an uploaded document into a generic industry explanation. Analyze the actual document first.\n"
        "5. When discussing a document, explicitly distinguish source statements from inference and external context.\n"
        "6. When a result is based on one security or one period, keep the conclusion scoped to that evidence.\n"
        "7. Do not describe two approaches as conflicting unless the evidence or the user's question establishes a conflict.\n"
        "8. If the user's question can be answered directly, do not ask an unnecessary follow-up question.\n"
        f"9. Allowed action types for this turn: {allowed}. This is a hard boundary.\n"
        "10. You do NOT have access to any tools in this synthesis phase. Do NOT attempt to run any tools or suggest code blocks that pretend to call tools.\n"
        "11. If a Bull vs. Bear debate was run, do NOT duplicate the Bull/Bear bullet points or the next steps in your text response since they are rendered in the card above. Instead, provide a short, high-level summary or professional concluding remarks (1-2 paragraphs).\n\n"
        "### TOOL EXECUTION RESULTS SUMMARY ###\n"
        f"{results_summary}\n\n"
        "Return a clear, faithful, professional answer. Educational simulation only; not official financial advice."
    )

    contents = []
    if chat_history:
        for msg in chat_history:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
    synthesis_prompt = f"User Prompt:\n{user_prompt}\n\n### WORKFLOW EXECUTION RESULTS ###\n{results_summary}"
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=synthesis_prompt)]))

    # Explicitly prevent synthesis layer from invoking tools by setting tools=None
    raw_response = generate_ai_response(
        contents,
        system_instruction,
        tools=None
    )
    
    validation_contract = contract or RequestContract(
        primary_goal=user_prompt[:100],
        requested_tasks=[],
        explicitly_prohibited=["suggest trade", "execute trade", "recommend trade", "run backtest"],
        source_required=bool(file_context),
        source_only=False,
        needs_tool=False,
        allowed_actions=allowed_actions,
        answer_scope="synthesis"
    )
    
    return validate_response(
        raw_response=raw_response,
        user_prompt=user_prompt,
        contract=validation_contract,
        file_context=file_context
    )
