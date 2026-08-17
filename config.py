import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Database Config
COCKROACH_DATABASE_URL = os.getenv("COCKROACH_DATABASE_URL")

# Gemini API Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Amazon S3 Storage Config
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "earnings-transcripts")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")

# Tavily API Config (Optional fallback)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Alpaca Paper Trading Config
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "True").lower() in ("true", "1", "yes")

# Default AI Model Settings

EMBEDDING_MODEL = "gemini-embedding-001"
GENERATIVE_MODEL = "gemini-3.1-flash-lite"
BATCH_EMBEDDING_SIZE = 100
DIRECT_CONTEXT_TOKEN_THRESHOLD = 1500

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
if not AWS_ACCESS_KEY_ID:
    missing_vars.append("AWS_ACCESS_KEY_ID")
if not AWS_SECRET_ACCESS_KEY:
    missing_vars.append("AWS_SECRET_ACCESS_KEY")
if not AWS_REGION:
    missing_vars.append("AWS_REGION")
if not AWS_S3_BUCKET:
    missing_vars.append("AWS_S3_BUCKET")

if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}", file=sys.stderr)
    print("Please copy .env.example to .env and populate the required keys.", file=sys.stderr)
    # We don't exit here immediately to allow tools/setup commands to run, but we will prevent full startup
