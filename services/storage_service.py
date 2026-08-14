import services.database as database
import io
import pypdf
import hashlib
import json
import threading
import pandas as pd
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential
import config

_supabase_client = None
_active_jobs_lock = threading.Lock()
_active_jobs = set()

def get_supabase_client():
    global _supabase_client
    if _supabase_client is None:
        if not config.SUPABASE_URL or not config.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables.")
        _supabase_client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    return _supabase_client

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True
)
def ensure_bucket_exists(bucket_name: str = "earnings-transcripts"):
    """Validates if the target Supabase storage bucket is active, creating it if needed."""
    client = get_supabase_client()
    try:
        buckets = client.storage.list_buckets()
        exists = any(b.name == bucket_name for b in buckets)
        if not exists:
            print(f"Bucket '{bucket_name}' not found. Attempting to create bucket...")
            client.storage.create_bucket(bucket_name, options={"public": False})
            print(f"Bucket '{bucket_name}' created successfully.")
    except Exception as e:
        print(f"Warning: Could not list/create storage bucket '{bucket_name}': {e}")
        print("Assuming bucket exists or permissions will handle it on upload.")

def upload_pdf_to_supabase(file_name: str, file_data: bytes, bucket_name: str = "earnings-transcripts") -> str:
    """Uploads PDF data to Supabase Storage and returns the file storage path."""
    clean_name = file_name.replace(" ", "_")
    try:
        client = get_supabase_client()
        ensure_bucket_exists(bucket_name)
        
        print(f"Uploading file '{clean_name}' to Supabase bucket '{bucket_name}'...")
        
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=5),
            reraise=True
        )
        def _do_upload():
            client.storage.from_(bucket_name).upload(
                path=clean_name,
                file=file_data,
                file_options={"upsert": "true", "content-type": "application/pdf"}
            )
        
        _do_upload()
        return f"supabase://{bucket_name}/{clean_name}"
    except Exception as e:
        print(f"Warning: Supabase upload failed due to storage policies or connectivity: {e}")
        print("Falling back to local reference.")
        return f"local://{clean_name}"

def extract_text_by_page(file_data: bytes) -> list[dict]:
    """Extracts text page-by-page from binary PDF data."""
    pdf_file = io.BytesIO(file_data)
    reader = pypdf.PdfReader(pdf_file)
    pages = []
    
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({
            "page_number": idx + 1,
            "text": text
        })
    return pages

def split_text_recursively(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Recursively splits text into narrative-rich semantic chunks at paragraph and sentence boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        if len(para) > chunk_size:
            sentences = para.split(". ")
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if current_len + len(sent) > chunk_size and current_chunk:
                    chunks.append(". ".join(current_chunk) + ".")
                    current_chunk = [current_chunk[-1]] if len(current_chunk) > 1 else []
                    current_len = sum(len(s) for s in current_chunk)
                current_chunk.append(sent)
                current_len += len(sent)
        else:
            if current_len + len(para) > chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [current_chunk[-1]] if len(current_chunk) > 1 else []
                current_len = sum(len(p) for p in current_chunk)
            current_chunk.append(para)
            current_len += len(para)
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

def process_pdf_into_chunks(file_data: bytes, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """Parses a PDF and returns chunks with enriched page metadata."""
    pages = extract_text_by_page(file_data)
    processed_chunks = []
    chunk_index = 0
    
    for p in pages:
        page_num = p["page_number"]
        page_text = p["text"].strip()
        if not page_text:
            continue
            
        page_chunks = split_text_recursively(page_text, chunk_size, chunk_overlap)
        
        for c_text in page_chunks:
            processed_chunks.append({
                "chunk_index": chunk_index,
                "chunk_text": c_text,
                "chunk_metadata": {
                    "page_number": page_num,
                    "char_count": len(c_text)
                }
            })
            chunk_index += 1
            
    return processed_chunks

def ingest_portfolio_csv(
    file_name: str, 
    file_data: bytes, 
    user_prompt: str = "",
    ticker_col_override: Optional[str] = None,
    shares_col_override: Optional[str] = None,
    cost_col_override: Optional[str] = None
) -> dict:
    """
    Parses a portfolio CSV file using pandas, maps columns, and returns parsed holdings.
    """
    df = pd.read_csv(io.BytesIO(file_data))
    
    # Normalize column headers
    cols = {c.lower().replace(" ", "").replace("_", "").replace("-", ""): c for c in df.columns}
    
    # Helper to find column matching a string case-insensitively and stripped
    def find_col(name: str):
        if not name:
            return None
        target = name.lower().replace(" ", "").replace("_", "").replace("-", "")
        if target in cols:
            return cols[target]
        for c in df.columns:
            if c.lower().replace(" ", "").replace("_", "").replace("-", "") == target:
                return c
        return None

    ticker_col = find_col(ticker_col_override)
    shares_col = find_col(shares_col_override)
    cost_col = find_col(cost_col_override)
    
    # Fallback to automatic detection if overrides not specified or not found
    if not ticker_col:
        for val in ["ticker", "symbol", "asset", "code"]:
            if val in cols:
                ticker_col = cols[val]
                break
            
    if not shares_col:
        for val in ["shares", "qty", "quantity", "volume", "amount"]:
            if val in cols:
                shares_col = cols[val]
                break
            
    if not cost_col:
        for val in ["costbasis", "avgprice", "averageprice", "cost", "purchaseprice", "price"]:
            if val in cols:
                cost_col = cols[val]
                break
            
    # Simple index fallback if naming doesn't match
    if not ticker_col or not shares_col or not cost_col:
        if len(df.columns) >= 3:
            ticker_col = df.columns[0]
            shares_col = df.columns[1]
            cost_col = df.columns[2]
            
    if not ticker_col or not shares_col or not cost_col:
        raise ValueError("Could not automatically map columns. Ensure CSV has Ticker, Shares, and Cost Basis headers.")
        
    parsed_holdings = []
    for _, row in df.iterrows():
        try:
            ticker = str(row[ticker_col]).upper().strip()
            shares = float(row[shares_col])
            cost_basis = float(row[cost_col])
            if ticker and not pd.isna(shares) and not pd.isna(cost_basis):
                parsed_holdings.append({
                    "ticker": ticker,
                    "shares": shares,
                    "cost_basis": cost_basis
                })
        except Exception:
            continue
            
    if not parsed_holdings:
        raise ValueError("No valid holdings could be parsed from the CSV.")
        
    overwrite_intent = False
    prompt_lower = user_prompt.lower()
    if any(k in prompt_lower for k in ["overwrite", "update", "replace", "set portfolio", "import", "load portfolio"]):
        overwrite_intent = True
        
    return {
        "holdings": parsed_holdings,
        "overwrite_intent": overwrite_intent
    }

def extract_strategies_from_ips(file_name: str, doc_text: str) -> list[str]:
    """
    Leverages Gemini to read the provided Investment Policy Statement (IPS) text and
    extract distinct, concrete qualitative investment strategy rules or guidelines.
    Returns them as a list of strings for review.
    """
    from agent.orchestrator import generate_ai_response
    
    print(f"Extracting strategy rules from IPS PDF '{file_name}'...")
    
    system_instruction = (
        "You are an expert financial compliance analyst. Your job is to analyze "
        "the provided Investment Policy Statement (IPS) text and extract distinct, concrete "
        "qualitative investment strategy rules or guidelines.\n\n"
        "Rules should look like:\n"
        "- 'Limit technology sector exposure to a maximum of 40% of total portfolio value.'\n"
        "- 'Trim holdings in a single stock if its weight exceeds 15%.'\n"
        "- 'Maintain at least 10% in liquid cash or short-term bonds.'\n\n"
        "Extract ONLY the rules as a raw JSON list of strings (e.g. [\"Rule 1\", \"Rule 2\"]). "
        "Do not include markdown formatting or explanations."
    )
    
    prompt = f"IPS Document Name: {file_name}\n\nContent:\n{doc_text[:12000]}"
    
    try:
        response_text = generate_ai_response(prompt, system_instruction=system_instruction)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        response_text = response_text.strip()
        
        rules = json.loads(response_text)
        if isinstance(rules, list):
            return [r.strip() for r in rules if r.strip()]
    except Exception as e:
        print(f"Failed to extract strategies from IPS: {e}")
    return []

def _run_ingestion_job_thread(job_id: str, file_name: str, file_data: bytes, user_prompt: str, ticker: str = None):
    """Internal target for background worker thread."""
    try:
        database.update_ingestion_job_status(job_id, 'extracting', 10)
        
        # 1. Compute SHA-256 Hash
        hasher = hashlib.sha256()
        hasher.update(file_data)
        file_hash = hasher.hexdigest()
        
        # Check duplicate
        existing_doc = database.get_document_by_hash(file_hash)
        if existing_doc:
            print(f"File '{file_name}' matches existing document hash. Reusing indexed document chunks.")
            database.update_ingestion_job_status(job_id, 'completed', 100)
            return
            
        file_type = "csv" if file_name.lower().endswith(".csv") else "pdf"
        
        # 2. Upload raw file to Supabase Storage
        database.update_ingestion_job_status(job_id, 'extracting', 25)
        storage_path = ""
        if file_type == "pdf":
            storage_path = upload_pdf_to_supabase(file_name, file_data)
        
        # Create Document entry
        doc_id = database.create_document(file_name, file_type, file_hash, storage_path)
        
        # 3. Branching based on Ingestion Type
        if file_type == "csv":
            # CSV Ingestion Case
            database.update_ingestion_job_status(job_id, 'chunking', 50)
            res = ingest_portfolio_csv(file_name, file_data, user_prompt)
            job_metadata = {
                "holdings": res["holdings"],
                "overwrite_intent": res["overwrite_intent"]
            }
            
            # Persist CSV portfolio holdings details text in document_chunks for RAG capability
            try:
                csv_text_summary = f"Portfolio holdings from CSV file '{file_name}':\n"
                for h in res["holdings"]:
                    csv_text_summary += f"- Ticker: {h['ticker']}, Shares: {h['shares']}, Cost Basis: ${h['cost_basis']}\n"
                
                csv_chunk = {
                    "chunk_index": 0,
                    "chunk_text": csv_text_summary,
                    "chunk_metadata": {
                        "document_id": doc_id,
                        "page_number": 1
                    }
                }
                
                from services.vector_store import get_embedding_provider
                embed_provider = get_embedding_provider()
                csv_chunk["embedding"] = embed_provider.get_embedding(csv_text_summary)
                
                database.save_document_chunks_batch(doc_id, None, [csv_chunk])
                print(f"  [+] Saved CSV holdings summary to vector space for document ID {doc_id}.")
            except Exception as e:
                print(f"Warning: Failed to save CSV chunks to vector space: {e}")
                
            database.update_ingestion_job_status(
                job_id, 
                'completed' if not res["overwrite_intent"] else 'persisting', 
                100, 
                error_message=json.dumps(job_metadata)
            )
            
        else:
            # PDF Ingestion Cases
            database.update_ingestion_job_status(job_id, 'chunking', 40)
            chunks = process_pdf_into_chunks(file_data, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            
            if not chunks:
                raise ValueError("No text extracted or chunked from PDF.")
                
            # Embeddings step
            database.update_ingestion_job_status(job_id, 'embedding', 60)
            from services.vector_store import get_embedding_provider
            embed_provider = get_embedding_provider()
            
            # Batch embedding generation
            chunk_texts = [c["chunk_text"] for c in chunks]
            batch_size = config.BATCH_EMBEDDING_SIZE
            all_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i+batch_size]
                pct = 60 + int((i / len(chunk_texts)) * 25)
                database.update_ingestion_job_status(job_id, 'embedding', pct)
                embeddings = embed_provider.get_embeddings_batch(batch_texts)
                all_embeddings.extend(embeddings)
                
            for idx, chunk in enumerate(chunks):
                chunk["embedding"] = all_embeddings[idx]
                
            database.update_ingestion_job_status(job_id, 'persisting', 90)
            
            # Batch DB insert
            database.save_document_chunks_batch(doc_id, ticker, chunks)
            
            # Strategy IPS check
            is_ips = "ips" in file_name.lower() or "investment policy" in file_name.lower()
            job_metadata = {}
            if is_ips:
                full_text = "\n".join([c["chunk_text"] for c in chunks])
                rules = extract_strategies_from_ips(file_name, full_text)
                if rules:
                    job_metadata["rules"] = rules
                
            database.update_ingestion_job_status(
                job_id, 
                'completed', 
                100,
                error_message=json.dumps(job_metadata) if job_metadata else None
            )
            
    except Exception as e:
        print(f"Error in background ingestion worker: {e}")
        database.update_ingestion_job_status(job_id, 'failed', 100, error_message=str(e))
    finally:
        with _active_jobs_lock:
            _active_jobs.discard(job_id)

def start_ingestion_job(job_id: str, file_name: str, file_data: bytes, user_prompt: str = "", ticker: str = None):
    """Enforces thread-safe duplicate check and starts a background thread to process the ingestion job."""
    with _active_jobs_lock:
        if job_id in _active_jobs:
            print(f"Worker for job '{job_id}' is already running. Skipping execution start.")
            return
        _active_jobs.add(job_id)
        
    # Launch worker thread
    t = threading.Thread(
        target=_run_ingestion_job_thread,
        args=(job_id, file_name, file_data, user_prompt, ticker),
        daemon=True
    )
    t.start()

def ingest_pdf_transcript(file_name: str, file_data: bytes, ticker: str):
    """Legacy interface maintained for backward compatibility inside test suites, running synchronously."""
    # We construct a mock job ID and process synchronously
    job_id = database.create_ingestion_job(file_name, "pdf")
    _run_ingestion_job_thread(job_id, file_name, file_data, "", ticker)
