import json
import concurrent.futures
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
import config
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
def generate_ai_response(prompt: str, system_instruction: str = None) -> str:
    client = get_gemini_client()
    
    # We construct config for the request if a system instruction is provided
    req_config = None
    if system_instruction:
        req_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.4
        )
    else:
        req_config = types.GenerateContentConfig(
            temperature=0.4
        )
        
    response = client.models.generate_content(
        model=config.GENERATIVE_MODEL,
        contents=prompt,
        config=req_config
    )
    return response.text

class RouterOutput(BaseModel):
    explanation: str = Field(
        description="Brief sentence explaining what the user wants and how we will route it. Keep it concise."
    )
    actions: List[ActionItem] = Field(
        description="The sequential list of actions to execute."
    )

@llm_retry
def route_user_intent(user_prompt: str, has_uploaded_file: bool, file_context: str = None) -> RouterOutput:
    client = get_gemini_client()
    
    system_instruction = (
        "You are an AI Routing Assistant for a financial system. "
        "Your task is to parse the user's prompt and decompose it into a structured plan "
        "of actions to run. Supported actions:\n"
        "1. 'debate': Ticker research scan (requires a stock ticker symbol like MSFT, AAPL, etc.).\n"
        "2. 'stress_test': Macro risk stress test (requires a macro scenario description like rate hikes, oil price drops, etc.).\n"
        "3. 'ingest': PDF transcript uploading (requires a stock ticker symbol. Note: This action should only be triggered if a file has been uploaded, as indicated by the has_uploaded_file context).\n"
        "4. 'performance_analysis': Calculate and return total portfolio valuation, daily price changes, gains, losses, or historical value trends.\n"
        "5. 'add_strategy': Add a new qualitative investment strategy guideline (requires strategy_text, e.g. 'Limit technology exposure to 40%'). Keep strategy_text concise.\n"
        "6. 'delete_strategy': Delete/remove an existing qualitative strategy guideline (requires strategy_target, which must strictly be the index/number or a short 2-3 word keyword of the rule to delete).\n"
        "7. 'update_strategy': Update/modify an existing strategy guideline (requires strategy_target which must strictly be the index/number or a short 2-3 word keyword, and strategy_text for the new wording).\n"
        "8. 'list_strategies': Show or list all currently configured strategy guidelines.\n"
        "If the user asks to add, delete, update, modify, or list strategies, you must route them to the corresponding strategy action instead of 'none'.\n"
        "For strategy_text, you must strictly extract the new wording based on the user's prompt. Do NOT hallucinate, change, or substitute any numbers, percentages, or specifics mentioned by the user (e.g. if the user says 'to 20 percent', your strategy_text must reflect '20 percent' or '20%', not '15%').\n"
        "Ensure all action properties (strategy_text, strategy_target, explanation) are clean, extremely concise, and contain absolutely no repetitive or looping text sequences.\n"
        "If a prompt is conversational or none of these actions fit, output 'none' as action_type.\n"
        "If a user asks to run multiple actions, output them in the sequential order they should execute."
    )
    
    prompt = (
        f"User Prompt: \"{user_prompt}\"\n"
        f"Has Uploaded File Context: {has_uploaded_file}\n"
    )
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
        return RouterOutput.model_validate_json(response.text)
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
def generate_conversational_response(user_prompt: str, strategies_str: str, file_context: str = None) -> str:
    system_instruction = (
        "You are MarketPulse AI, a professional financial assistant. "
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
        
    return generate_ai_response(prompt, system_instruction)

@llm_retry
def synthesize_chat_response(user_prompt: str, results_summary: str, strategies_str: str = "", file_context: str = None) -> str:
    system_instruction = (
        "You are a friendly, highly professional AI Investment Assistant named MarketPulse AI. "
        "You have executed tools (debates, macro stress tests, document ingestion, or strategy modifications) to satisfy the user's request. "
        "Your goal is to synthesize the outcomes of these tools into a clean, action-oriented, and "
        "insightful conversational response. Always check these outcomes against the user's qualitative strategy rules "
        "listed below, and emphasize how they align or conflict.\n\n"
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
        
    return generate_ai_response(prompt, system_instruction)

