import json
import concurrent.futures
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
import config
from .orchestrator import generate_ai_response

def generate_bull_perspective(ticker: str, holdings_str: str, news_str: str, strategy_str: str, docs_str: str) -> str:
    system_instruction = (
        "You are an expert, optimistic Wall Street Equity Analyst and Growth Investor. "
        "Your task is to analyze the provided financial data and generate a compelling, data-driven "
        "BULL case for the stock. Focus on upside catalysts, competitive advantages, future earnings growth, "
        "macro tailwinds, and why it aligns with the user's investment strategies. Do not make up figures."
    )
    
    prompt = f"""
    Analyze the following stock ticker: {ticker.upper()}
    
    ### QUANTITATIVE HOLDINGS INFO:
    {holdings_str}
    
    ### RECENT NEWS INGESTED:
    {news_str}
    
    ### USER STRATEGY RULES:
    {strategy_str}
    
    ### TRANSCRIPT/DOCUMENT SEMANTIC CONTEXT:
    {docs_str}
    
    Based on this, write a highly structured bullish thesis. Focus on:
    1. Key Growth Catalysts & Competitive Moats
    2. How this ticker aligns with user's investment principles
    3. Target indicators and positive signs to monitor
    """
    
    return generate_ai_response(prompt, system_instruction)

def generate_bear_perspective(ticker: str, holdings_str: str, news_str: str, strategy_str: str, docs_str: str) -> str:
    system_instruction = (
        "You are a conservative Chief Risk Officer (CRO) and Bearish Short-Seller. "
        "Your task is to analyze the provided financial data and generate a rigorous, skeptical "
        "BEAR case for the stock. Focus on operational headwinds, macro risks, concentration limits, "
        "financial structural vulnerabilities, and direct contradictions to the user's strategy rules. Do not make up figures."
    )
    
    prompt = f"""
    Analyze the following stock ticker: {ticker.upper()}
    
    ### QUANTITATIVE HOLDINGS INFO:
    {holdings_str}
    
    ### RECENT NEWS INGESTED:
    {news_str}
    
    ### USER STRATEGY RULES:
    {strategy_str}
    
    ### TRANSCRIPT/DOCUMENT SEMANTIC CONTEXT:
    {docs_str}
    
    Based on this, write a highly structured bearish thesis. Focus on:
    1. Threat Vectors, Operational Risks & Tail Risks
    2. Areas where the asset clashes with the user's strategy memory (e.g., concentration, valuation warnings)
    3. Warning triggers and metrics to watch closely
    """
    
    return generate_ai_response(prompt, system_instruction)

def synthesize_debate(ticker: str, bull_perspective: str, bear_perspective: str) -> str:
    system_instruction = (
        "You are the Chair of a Multi-Strategy Investment Committee. "
        "Your role is to compile and synthesize opposing Bull and Bear perspectives on an asset. "
        "Present a balanced, objective, and highly action-oriented executive summary that "
        "contrasts the bull and bear cases. Provide educational next steps for a human researcher. "
        "Emphasize safety and clarity. This is for research simulations only and is not trading advice."
    )
    
    prompt = f"""
    Compile a synthesis for ticker: {ticker.upper()}
    
    ### BULL PERSPECTIVE GENERATED:
    {bull_perspective}
    
    ### BEAR PERSPECTIVE GENERATED:
    {bear_perspective}
    
    Provide a unified report with:
    1. **Executive Research Summary**
    2. **Key Battlegrounds** (Where do the Bull and Bear argue the most?)
    3. **Strategy Alignment Flags** (Is this asset safe based on stored qualitative rules?)
    4. **Educational Research Next Steps** (What should a human check next?)
    """
    
    return generate_ai_response(prompt, system_instruction)

def run_macro_stress_test(scenario_prompt: str, holdings_str: str, strategies_str: str) -> str:
    system_instruction = (
        "You are a Senior Macroeconomic Risk Strategist. "
        "Your task is to evaluate user portfolio holdings against a hypothetical macro stress event. "
        "Analyze the transmission channels of the stress event to the stock holdings. "
        "Incorporate qualitative rules. Be analytical, educational, and safety-oriented. "
        "Provide qualitative simulation details, estimating the directional impact on portfolio risk and rule compliance."
    )
    
    prompt = f"""
    Analyze the impact of the following MACRO SCENARIO:
    "{scenario_prompt}"
    
    ### CURRENT PORTFOLIO HOLDINGS:
    {holdings_str}
    
    ### USER STRATEGY GUIDELINES:
    {strategies_str}
    
    Write a detailed Risk Report explaining:
    1. **Direct Transmission Channels**: How the macro event impacts the industries and tickers in the portfolio.
    2. **Portfolio Impact Assessment**: Directional risk warning (e.g., high risk, sector specific, interest rate sensitivity).
    3. **Rule Compliance Warnings**: Do any of the holdings risk violating user strategies under this scenario?
    4. **Mitigation Areas**: Strategic research suggestions to hedge or rebalance.
    """
    
    return generate_ai_response(prompt, system_instruction)

def run_parallel_debate(ticker: str, holdings_str: str, news_str: str, strategy_str: str, docs_str: str) -> dict:
    """Executes Bull and Bear generation in parallel threads to optimize latency."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_bull = executor.submit(
            generate_bull_perspective, ticker, holdings_str, news_str, strategy_str, docs_str
        )
        future_bear = executor.submit(
            generate_bear_perspective, ticker, holdings_str, news_str, strategy_str, docs_str
        )
        
        # Gather results (will block until both complete)
        bull = future_bull.result()
        bear = future_bear.result()
        
    return {
        "bull": bull,
        "bear": bear
    }

