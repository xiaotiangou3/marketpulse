import sys
import psycopg2
from google import genai
from google.genai import types
import boto3
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

def verify_s3():
    print("Testing Amazon S3 Connection...")
    if not config.AWS_ACCESS_KEY_ID or not config.AWS_SECRET_ACCESS_KEY:
        print("  [-] AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY is not set.")
        return False
    try:
        session = boto3.Session(
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )
        extra_kwargs = {}
        if config.AWS_S3_ENDPOINT_URL:
            extra_kwargs["endpoint_url"] = config.AWS_S3_ENDPOINT_URL
        s3 = session.client("s3", **extra_kwargs)
        s3.head_bucket(Bucket=config.AWS_S3_BUCKET)
        print(f"  [+] Amazon S3 Connected successfully! Bucket '{config.AWS_S3_BUCKET}' is accessible.")
        return True
    except Exception as e:
        print(f"  [-] Amazon S3 Connection failed: {e}")
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
    s3_success = verify_s3()
    tavily_success = verify_tavily()
    
    print("\n==================================================")
    if cr_success and gemini_success and s3_success and tavily_success:
        print("   ALL CONNECTIONS SUCCESSFUL! Ready to proceed.  ")
        print("==================================================")
        sys.exit(0)
    else:
        print("   SOME CONNECTIONS FAILED. Review errors above.  ")
        print("==================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
