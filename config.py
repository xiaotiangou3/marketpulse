import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Database Config
COCKROACH_DATABASE_URL = os.getenv("COCKROACH_DATABASE_URL")

# Gemini API Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Supabase Storage Config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Tavily API Config (Optional fallback)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Default AI Model Settings
EMBEDDING_MODEL = "gemini-embedding-001"
GENERATIVE_MODEL = "gemini-3.1-flash-lite"

# Document Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retention Policy
TTL_DAYS = 30

# Stock Price Polling Settings (in minutes)
PRICE_POLLING_INTERVAL_MINUTES = int(os.getenv("PRICE_POLLING_INTERVAL_MINUTES", "30"))

# Validation of critical variables
missing_vars = []
if not COCKROACH_DATABASE_URL:
    missing_vars.append("COCKROACH_DATABASE_URL")
if not GEMINI_API_KEY:
    missing_vars.append("GEMINI_API_KEY")
if not SUPABASE_URL:
    missing_vars.append("SUPABASE_URL")
if not SUPABASE_KEY:
    missing_vars.append("SUPABASE_KEY")

if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}", file=sys.stderr)
    print("Please copy .env.example to .env and populate the required keys.", file=sys.stderr)
    # We don't exit here immediately to allow tools/setup commands to run, but we will prevent full startup
