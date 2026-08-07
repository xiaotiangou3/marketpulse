import sys
import psycopg2
from google import genai
from google.genai import types
from supabase import create_client
from tavily import TavilyClient
import config

def verify_cockroachdb():
    print("Testing CockroachDB Connection...")
    if not config.COCKROACH_DATABASE_URL:
        print("  [-] COCKROACH_DATABASE_URL is not set.")
        return False
    try:
        conn = psycopg2.connect(config.COCKROACH_DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()[0]
        cur.close()
        conn.close()
        print(f"  [+] CockroachDB Connected successfully! Version: {db_version}")
        return True
    except Exception as e:
        print(f"  [-] CockroachDB Connection failed: {e}")
        return False

def verify_gemini():
    print("Testing Gemini API Connection...")
    if not config.GEMINI_API_KEY:
        print("  [-] GEMINI_API_KEY is not set.")
        return False
    try:
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents="Connection validation check.",
            config=types.EmbedContentConfig(
                output_dimensionality=768
            )
        )
        embedding = response.embeddings[0].values
        dims = len(embedding)
        print(f"  [+] Gemini API Connected successfully! Embedding generated with {dims} dimensions.")
        if dims != 768:
            print(f"  [!] Warning: Expected 768 dimensions for {config.EMBEDDING_MODEL}, got {dims}.")
        return True
    except Exception as e:
        print(f"  [-] Gemini API Connection failed: {e}")
        return False

def verify_supabase():
    print("Testing Supabase Connection...")
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        print("  [-] SUPABASE_URL or SUPABASE_KEY is not set.")
        return False
    try:
        supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        # Try listing buckets to check storage access
        buckets = supabase.storage.list_buckets()
        print(f"  [+] Supabase Storage Connected successfully! Found {len(buckets)} buckets.")
        return True
    except Exception as e:
        print(f"  [-] Supabase Connection failed: {e}")
        return False

def verify_tavily():
    print("Testing Tavily Search API Connection...")
    if not config.TAVILY_API_KEY:
        print("  [~] TAVILY_API_KEY is not set. Skipping Tavily check. (Macro queries will have no fallback).")
        return True
    try:
        tavily = TavilyClient(api_key=config.TAVILY_API_KEY)
        results = tavily.search(query="Microsoft stock news", max_results=1)
        print(f"  [+] Tavily Search API Connected successfully! Found {len(results.get('results', []))} test results.")
        return True
    except Exception as e:
        print(f"  [-] Tavily Search API Connection failed: {e}")
        return False

def main():
    print("==================================================")
    print("   MarketPulse AI Connection Verification Suite   ")
    print("==================================================\n")
    
    if config.missing_vars:
        print("CRITICAL: Some required environment variables are missing.")
        print("Please check your .env configuration.")
        sys.exit(1)
        
    cr_success = verify_cockroachdb()
    gemini_success = verify_gemini()
    supabase_success = verify_supabase()
    tavily_success = verify_tavily()
    
    print("\n==================================================")
    if cr_success and gemini_success and supabase_success and tavily_success:
        print("   ALL CONNECTIONS SUCCESSFUL! Ready to proceed.  ")
        print("==================================================")
        sys.exit(0)
    else:
        print("   SOME CONNECTIONS FAILED. Review errors above.  ")
        print("==================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
