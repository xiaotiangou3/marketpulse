import sys
import io
import database
import services
import providers
import agent

# ==========================================
# RUNTIME MOCK INJECTIONS (Bypasses AI API)
# ==========================================

class MockEmbeddingProvider(providers.EmbeddingProvider):
    def get_embedding(self, text: str) -> list[float]:
        # Returns a deterministic 768-dimensional mock vector
        return [0.05] * 768

# Override services embedding provider
services._embedding_provider = MockEmbeddingProvider()

# Mock generative AI responses to bypass rate limits
agent.run_parallel_debate = lambda ticker, holdings, news, strategy, docs: {
    "bull": "Mock Bull Case: MSFT has strong growth prospects and Azure adoption is high.",
    "bear": "Mock Bear Case: MSFT faces high valuation and macro headwind risks."
}
agent.synthesize_debate = lambda ticker, bull, bear: (
    "### Mock Synthesized Investment Report: MSFT\n\n"
    "1. Executive Summary: Both cases show strong elements. Ingested docs: Azure is accelerating.\n"
    "2. Strategy Alignment: High alignment."
)
agent.run_macro_stress_test = lambda scenario, holdings, strategies: (
    "### Mock Macro Stress Risk Report\n\n"
    f"Simulating scenario: {scenario}. Estimated direction: negative."
)

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

def main():
    print("==================================================")
    print("   MARKETPULSE DATABASE & SCHEMAS TEST SUITE      ")
    print("   (Mock Embeddings & Mock LLMs - Zero Quota Cost)")
    print("==================================================")
    
    try:
        # Step 1: Run migrations
        print("\n[Step 1] Running database migrations...")
        database.run_migrations()
        print("  [+] Migrations completed successfully.")
        
        # Step 2: Set up test holdings
        print("\n[Step 2] Adding test holdings...")
        msft_id = services.add_stock_holding("MSFT", 15.0, 310.0)
        aapl_id = services.add_stock_holding("AAPL", 25.0, 160.0)
        print(f"  [+] Added MSFT (15 shares @ $310) -> ID: {msft_id}")
        print(f"  [+] Added AAPL (25 shares @ $160) -> ID: {aapl_id}")
        
        # Step 3: Set up test investment strategy
        print("\n[Step 3] Saving qualitative strategy rule...")
        strategy_text = "Favor high cash-flow software stocks with strong enterprise pricing power."
        strat_id = services.save_investment_strategy(strategy_text)
        print(f"  [+] Strategy saved successfully -> ID: {strat_id}")
        
        # Step 4: PDF Ingestion & Semantic Vector Search
        print("\n[Step 4] Simulating PDF Earnings Transcript ingestion...")
        pdf_bytes = make_simple_pdf()
        pdf_name = "msft_q3_transcript_test.pdf"
        services.ingest_pdf_transcript(pdf_name, pdf_bytes, "MSFT")
        print("  [+] PDF Ingestion successfully triggered.")
        
        # Verify chunks exist and are searchable
        print("  Running semantic query against document chunks...")
        embed_provider = services.get_embedding_provider()
        query_text = "Azure and AI adoption cloud metrics"
        query_embed = embed_provider.get_embedding(query_text)
        matching_chunks = database.search_document_chunks_semantic("MSFT", query_embed, limit=2)
        print(f"  [+] Retrieved {len(matching_chunks)} chunks semantically:")
        for c in matching_chunks:
            print(f"    - Page {c['chunk_metadata'].get('page_number')}: '{c['chunk_text']}' (distance: {c['distance']:.3f})")
        
        if not matching_chunks:
            print("  [-] Error: No matching chunks found. Ingestion failed.")
            sys.exit(1)
            
        # Step 5: Run portfolio analysis and dual-agent debate
        print("\n[Step 5] Triggering portfolio research and debate scan for MSFT...")
        analysis_result = services.conduct_portfolio_analysis("MSFT")
        
        print("\n  ========================================")
        print("  [BULL PERSPECTIVE (MOCKED)]")
        print("  ========================================")
        print(analysis_result['bull'])
        
        print("\n  ========================================")
        print("  [BEAR PERSPECTIVE (MOCKED)]")
        print("  ========================================")
        print(analysis_result['bear'])
        
        print("\n  ========================================")
        print("  [SYNTHESIZED INVESTMENT REPORT (MOCKED)]")
        print("  ========================================")
        print(analysis_result['synthesis'])
        
        # Ensure that document context was actually parsed and loaded into the prompt
        if "Azure" not in analysis_result['docs_context']:
            print("  [-] Error: Document context was not retrieved during debate scan.")
            sys.exit(1)
        else:
            print("  [+] Verified: Ingested document was utilized in news scan context.")
            
        # Step 6: Run Macro Stress Test
        print("\n[Step 6] Triggering macro stress test scenario...")
        scenario = "Severe credit tightening triggers interest rate hikes of 100bps."
        stress_report = services.execute_stress_test(scenario)
        
        print("\n  ========================================")
        print("  [MACRO STRESS RISK REPORT (MOCKED)]")
        print("  ========================================")
        print(stress_report)
        
        # Step 7: Verify research logs
        print("\n[Step 7] Querying research audit logs...")
        logs = database.get_research_logs()
        print(f"  [+] Found {len(logs)} entries in audit history.")
        for idx, log in enumerate(logs[:2]):
            print(f"    Log {idx+1}: '{log['prompt_query']}'")
            print(f"      Distance score: {log['vector_distance']}")
            print(f"      Metadata logged: {log['session_metadata']}")
            
        # Verify that audit log has been successfully saved
        if len(logs) < 2:
            print("  [-] Error: Expected at least 2 audit logs (one debate, one stress test).")
            sys.exit(1)
            
        # Step 8: Clean up
        print("\n[Step 8] Cleaning up test holdings...")
        services.remove_stock_holding(msft_id)
        services.remove_stock_holding(aapl_id)
        print("  [+] Test assets removed.")
        
        print("\n==================================================")
        print("   DATABASE & SCHEMAS TEST COMPLETED SUCCESSFULLY!")
        print("==================================================")
        
    except Exception as e:
        print(f"\n[-] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
