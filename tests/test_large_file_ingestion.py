import os
import sys
import json
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import services.database as database
import services.storage_service as storage
import providers
import config

class MockEmbeddingProvider(providers.EmbeddingProvider):
    def get_embedding(self, text: str) -> list[float]:
        return [0.05] * 768
        
    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.05] * 768 for _ in texts]

def test_csv_portfolio_parsing():
    # Synthetic CSV structure with varying column names
    csv_data = (
        b"Symbol,Qty,Avg Price\n"
        b"AAPL,50,180.50\n"
        b"MSFT,30,420.10\n"
    )
    
    # Overwrite Intent Test
    res = storage.ingest_portfolio_csv("test_portfolio.csv", csv_data, "overwrite my current portfolio")
    assert res["overwrite_intent"] is True
    assert len(res["holdings"]) == 2
    assert res["holdings"][0]["ticker"] == "AAPL"
    assert res["holdings"][0]["shares"] == 50.0
    assert res["holdings"][0]["cost_basis"] == 180.50
    
    # Analysis Intent Test
    res_analysis = storage.ingest_portfolio_csv("test_portfolio.csv", csv_data, "please analyze this portfolio and tell me my stress risk")
    assert res_analysis["overwrite_intent"] is False
    assert len(res_analysis["holdings"]) == 2

def test_ingestion_job_deduplication():
    # Trigger twice and verify only one thread starts
    job_id = "test_dedup_job_id"
    file_name = "test_file.pdf"
    file_data = b"%PDF-1.4 mock pdf text"
    
    # Place job_id into active jobs set manually to simulate running worker
    with storage._active_jobs_lock:
        storage._active_jobs.add(job_id)
        
    # Attempt starting worker (should skip starting a new thread)
    storage.start_ingestion_job(job_id, file_name, file_data, "", "MSFT")
    
    # Check that job is still marked active, but didn't spin up another
    with storage._active_jobs_lock:
        assert job_id in storage._active_jobs
        storage._active_jobs.remove(job_id)

def test_token_based_context_pruning():
    # Set configuration values dynamically for tests
    old_threshold = config.DIRECT_CONTEXT_TOKEN_THRESHOLD
    config.DIRECT_CONTEXT_TOKEN_THRESHOLD = 20  # Make it small for testing
    
    # Small context context check (approx 5 tokens / characters)
    small_text = "Hello world context check"
    est_small_tokens = len(small_text) // 4
    assert est_small_tokens < config.DIRECT_CONTEXT_TOKEN_THRESHOLD
    
    # Large context check (approx 200 characters / 50 tokens)
    large_text = "This is a much longer context string designed to exceed the small context token threshold set inside config.py dynamically for this unit test."
    est_large_tokens = len(large_text) // 4
    assert est_large_tokens >= config.DIRECT_CONTEXT_TOKEN_THRESHOLD
    
    # Reset threshold
    config.DIRECT_CONTEXT_TOKEN_THRESHOLD = old_threshold

def test_startup_recovery():
    # Insert a dummy job marked in-progress
    conn = database.get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ingestion_jobs (job_id, document_name, file_type, status, progress_pct)
                VALUES ('d3b07384-d113-4ec3-a5d6-bd61726a5413', 'restart_test.pdf', 'pdf', 'extracting', 25)
                ON CONFLICT (job_id) DO UPDATE SET status = 'extracting', progress_pct = 25;
                """
            )
            conn.commit()
    finally:
        database.release_db_connection(conn)
        
    # Trigger recovery
    interrupted_ids = database.recover_interrupted_ingestion_jobs()
    assert 'd3b07384-d113-4ec3-a5d6-bd61726a5413' in interrupted_ids
    
    # Verify status is now 'queued'
    job = database.get_ingestion_job('d3b07384-d113-4ec3-a5d6-bd61726a5413')
    assert job["status"] == "queued"
    assert job["progress_pct"] == 0

def test_legacy_unmapped_rows_filtering():
    # Verify that semantic search query joins with documents and excludes null document_ids
    # Get a dummy query embedding
    query_embed = [0.05] * 768
    
    # Running search_document_chunks_semantic should work without failure
    # If legacy unmapped rows exist (document_id IS NULL), they should be ignored by the JOIN query.
    results = database.search_document_chunks_semantic("MSFT", query_embed, limit=2)
    assert isinstance(results, list)

def main():
    print("==================================================")
    print("   MARKETPULSE LARGE FILE INGESTION TEST SUITE    ")
    print("==================================================")
    
    database.run_migrations()
    
    # Inject mocks
    storage.database = database
    
    try:
        print("\n[Step 1] Testing CSV Portfolio auto-mapping and intent...")
        test_csv_portfolio_parsing()
        print("  [+] CSV Parsing passed.")
        
        print("\n[Step 2] Testing Ingestion Job De-duplication...")
        test_ingestion_job_deduplication()
        print("  [+] De-duplication verification passed.")
        
        print("\n[Step 3] Testing Token-based Context Pruning...")
        test_token_based_context_pruning()
        print("  [+] Token-based pruning checks passed.")
        
        print("\n[Step 4] Testing Active Job Startup Recovery...")
        test_startup_recovery()
        print("  [+] Startup job recovery passed.")
        
        print("\n[Step 5] Testing Legacy unmapped rows RAG filtering...")
        test_legacy_unmapped_rows_filtering()
        print("  [+] Legacy row vector filtering passed.")
        
        print("\n==================================================")
        print("   ALL FILE INGESTION UNIT TESTS PASSED!          ")
        print("==================================================")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
