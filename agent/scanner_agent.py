import json
import concurrent.futures
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
import config

llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)

class ActionItem(BaseModel):
    action_type: str = Field(
        description="Must be one of: 'debate', 'stress_test', 'ingest', 'performance_analysis', 'add_strategy', 'delete_strategy', 'update_strategy', 'list_strategies', or 'none'"
    )
    ticker: Optional[str] = Field(
        None, 
        description="Ticker symbol (capitalized, e.g. 'MSFT') if action_type is 'debate' or 'ingest'"
    )
    scenario: Optional[str] = Field(
        None, 
        description="Scenario description if action_type is 'stress_test'"
    )
    strategy_text: Optional[str] = Field(
        None,
        description="New or updated strategy text if action_type is 'add_strategy' or 'update_strategy'. Keep it concise and avoid repeating phrases."
    )
    strategy_target: Optional[str] = Field(
        None,
        description="Keep it extremely short: strictly either a rule number (e.g. '1') or a short 2-3 word keyword (e.g. 'tech limit'). Do not write long explanations or repeat phrases."
    )

@llm_retry
def generate_news_suggestions(news_title: str, news_summary: str, holdings_str: str, strategies_str: str) -> str:
    system_instruction = (
        "You are MarketPulse AI, a professional financial research sentinel. "
        "Your task is to analyze a specific market news article and provide actionable next-step suggestions "
        "tailored to the user's specific stock holdings and stored qualitative strategy rules.\n\n"
        "Directly evaluate if the news triggers any of the strategy rules, presents risks, "
        "or creates potential opportunities for their holdings. "
        "Be analytical, precise, and educational. Avoid generic summaries; focus purely on the strategic implications."
    )
    
    prompt = f"""
    Analyze this news article:
    **Title**: {news_title}
    **Summary**: {news_summary}
    
    ### USER'S PORTFOLIO HOLDINGS:
    {holdings_str}
    
    ### USER'S STRATEGY GUIDELINES:
    {strategies_str}
    
    Based on the news and the user's holdings/strategies, write a concise risk and action suggestions report.
    Include:
    1. **Strategic Impact**: Does this news directly impact the user's holdings, and does it align or clash with their strategy rules?
    2. **Suggested Actions**: Clear next steps (e.g. 'Monitor Q4 margins', 'No action needed', 'Consider rebalancing tech weight if NVDA surges').
    3. **Rule Compliance Flags**: Highlight any specific strategy rule index or keyword that is triggered or violated.
    """
    
    return generate_ai_response(prompt, system_instruction)

