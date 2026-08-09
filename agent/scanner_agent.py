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

from .orchestrator import ActionItem, generate_ai_response

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

