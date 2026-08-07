import sys
import database
import services
import providers
import agent

def make_simple_pdf():
    # A basic valid PDF structure containing a single page of text
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj\n"
        b"<</Type /Catalog /Pages 2 0 R>>\n"
        b"endobj\n"
        b"2 0 obj\n"
        b"<</Type /Pages /Kids [3 0 R] /Count 1>>\n"
        b"endobj\n"
        b"3 0 obj\n"
        b"<</Type /Page\n"
        b"  /Parent 2 0 R\n"
        b"  /MediaBox [0 0 595 842]\n"
        b"  /Resources <<\n"
        b"    /Font <<\n"
        b"      /F1 <<\n"
        b"        /Type /Font\n"
        b"        /Subtype /Type1\n"
        b"        /BaseFont /Helvetica\n"
        b"      >>\n"
        b"    >>\n"
        b"  >>\n"
        b"  /Contents 4 0 R\n"
        b">>\n"
        b"endobj\n"
        b"4 0 obj\n"
        b"<</Length 124>>\n"
        b"stream\n"
        b"BT\n"
        b"  /F1 12 Tf\n"
        b"  72 712 Td\n"
        b"  (Microsoft Q3 Earnings Transcript: Cloud growth is accelerating.) Tj\n"
        b"  0 -20 Td\n"
        b"  (We expect Azure growth of 30 percent next quarter due to strong AI adoption.) Tj\n"
        b"ET\n"
        b"endstream\n"
        b"endobj\n"
        b"xref\n"
        b"0 5\n"
        b"0000000000 65535 f\n"
        b"0000000009 00000 n\n"
        b"0000000058 00000 n\n"
        b"0000000115 00000 n\n"
        b"0000000329 00000 n\n"
        b"trailer\n"
        b"<</Size 5 /Root 1 0 R>>\n"
        b"startxref\n"
        b"504\n"
        b"%%EOF\n"
    )
    return pdf_content

# ==========================================
# RUNTIME MOCK INJECTIONS (Bypasses AI API)
# ==========================================

class MockEmbeddingProvider(providers.EmbeddingProvider):
    def get_embedding(self, text: str) -> list[float]:
        return [0.05] * 768

# Override services embedding provider
services._embedding_provider = MockEmbeddingProvider()

# We mock agent.route_user_intent to return different structured actions based on prompt keywords
def mock_route_user_intent(user_prompt: str, has_uploaded_file: bool) -> agent.RouterOutput:
    user_prompt_lower = user_prompt.lower()
    actions = []
    explanation = f"Mock routed user prompt: '{user_prompt}'"
    
    if "debate" in user_prompt_lower or "review" in user_prompt_lower:
        ticker = "MSFT"
        if "aapl" in user_prompt_lower:
            ticker = "AAPL"
        actions.append(agent.ActionItem(action_type="debate", ticker=ticker))
        
    if "stress" in user_prompt_lower or "rate hike" in user_prompt_lower:
        actions.append(agent.ActionItem(action_type="stress_test", scenario="Interest rates rise 50bps."))
        
    if "ingest" in user_prompt_lower or "upload" in user_prompt_lower:
        ticker = "MSFT"
        if "aapl" in user_prompt_lower:
            ticker = "AAPL"
        actions.append(agent.ActionItem(action_type="ingest", ticker=ticker))
        
    if not actions:
        actions.append(agent.ActionItem(action_type="none"))
        
    return agent.RouterOutput(explanation=explanation, actions=actions)

agent.route_user_intent = mock_route_user_intent

# Mock synthesis response
def mock_synthesize_chat_response(user_prompt: str, results_summary: str, *args, **kwargs) -> str:
    return (
        f"### MarketPulse AI Assistant Response\n\n"
        f"I processed your prompt: \"{user_prompt}\" and executed the requested research tools.\n\n"
        f"**Summary of Tool Executed Findings:**\n{results_summary}"
    )

agent.synthesize_chat_response = mock_synthesize_chat_response

# Mock direct LLM call for conversational prompts
def mock_generate_ai_response(prompt: str, system_instruction: str = None) -> str:
    return "Hello! I am MarketPulse AI. How can I help you research your portfolio today?"

agent.generate_ai_response = mock_generate_ai_response

def main():
    print("==================================================")
    print("   MARKETPULSE AI CHATBOT ROUTING TEST SUITE      ")
    print("   (Mock API Calls - Local Database verification)")
    print("==================================================")
    
    try:
        # Step 1: Run migrations
        print("\n[Step 1] Running database migrations...")
        database.run_migrations()
        print("  [+] Migrations completed successfully.")
        
        # Step 2: Set up test holdings & strategies
        print("\n[Step 2] Setting up holdings and strategy rules...")
        msft_id = services.add_stock_holding("MSFT", 10.0, 320.0)
        strat_id = services.save_investment_strategy("Focus on high margins and low volatility.")
        print(f"  [+] Added MSFT holding -> ID: {msft_id}")
        print(f"  [+] Saved strategy rule -> ID: {strat_id}")
        
        # Step 3: Test routing single action: Debate
        print("\n[Step 3] Testing chatbot prompt: 'Run a debate on MSFT'...")
        res = services.run_chatbot_session("Run a debate on MSFT")
        print(f"  [+] Chatbot Response:\n{res['response']}")
        print(f"  [+] Actions resolved: {res['router']['actions']}")
        assert len(res['actions_run']) == 1
        assert res['actions_run'][0]['type'] == "debate"
        
        # Step 4: Test routing multiple actions: Debate + Stress Test
        print("\n[Step 4] Testing chatbot prompt: 'Run a debate on MSFT and do a rate hike stress test'...")
        res = services.run_chatbot_session("Run a debate on MSFT and do a rate hike stress test")
        print(f"  [+] Chatbot Response:\n{res['response']}")
        print(f"  [+] Actions resolved: {res['router']['actions']}")
        assert len(res['actions_run']) == 2
        assert res['actions_run'][0]['type'] == "debate"
        assert res['actions_run'][1]['type'] == "stress_test"
        
        # Step 5: Test routing Ingestion without file
        print("\n[Step 5] Testing chatbot prompt: 'Ingest transcript for MSFT' (without file uploaded)...")
        res = services.run_chatbot_session("Ingest transcript for MSFT")
        print(f"  [+] Chatbot Response:\n{res['response']}")
        print(f"  [+] Actions resolved: {res['router']['actions']}")
        assert res['actions_run'][0]['status'] == "missing_file"
        
        # Step 6: Test routing Ingestion with file bytes
        print("\n[Step 6] Testing chatbot prompt: 'Ingest transcript for MSFT' (with mock file uploaded)...")
        mock_pdf = make_simple_pdf()
        res = services.run_chatbot_session("Ingest transcript for MSFT", "test_transcript.pdf", mock_pdf)
        print(f"  [+] Chatbot Response:\n{res['response']}")
        print(f"  [+] Actions resolved: {res['router']['actions']}")
        # The upload will warn about Supabase RLS and proceed to process locally
        assert res['actions_run'][0]['status'] == "success"
        
        # Step 7: Test conversational fallback
        print("\n[Step 7] Testing conversational fallback prompt: 'Hello MarketPulse!'...")
        res = services.run_chatbot_session("Hello MarketPulse!")
        print(f"  [+] Chatbot Response:\n{res['response']}")
        assert res['actions_run'][0]['type'] == "conversational"
        
        # Step 8: Clean up
        print("\n[Step 8] Cleaning up holdings...")
        services.remove_stock_holding(msft_id)
        print("  [+] Clean up completed.")
        
        print("\n==================================================")
        print("   CHATBOT ORCHESTRATION TEST COMPLETED SUCCESSFULLY!")
        print("==================================================")
        
    except Exception as e:
        print(f"\n[-] Chatbot integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
