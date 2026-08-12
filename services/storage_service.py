import services.database as database
import io
import pypdf
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential
import config
_supabase_client = None

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
        # Try ensuring bucket exists
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
        print("Falling back to local reference. PDF text parsing and CockroachDB vector indexing will still proceed.")
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
                    "char_count": len(c_text),
                    "source": "earnings_transcript"
                }
            })
            chunk_index += 1
            
    return processed_chunks

def ingest_pdf_transcript(file_name: str, file_data: bytes, ticker: str):
    """
    Orchestrates the ingestion, chunking, embedding, and database storage of a PDF transcript.
    """
    ticker = ticker.upper().strip()
    print(f"Ingesting PDF transcript '{file_name}' for {ticker}...")
    
    # 1. Upload to Supabase Storage
    storage_path = upload_pdf_to_supabase(file_name, file_data)
    print(f"  [+] Uploaded to Supabase Storage: {storage_path}")
    
    # 2. Extract and split text recursively
    chunks = process_pdf_into_chunks(file_data, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    print(f"  [+] Processed PDF into {len(chunks)} chunks.")
    
    if not chunks:
        print("  [!] Warning: No text extracted or chunked from PDF. Skipping database insert.")
        return
        
    # 3. Generate embeddings for each chunk
    from .vector_store import get_embedding_provider
    embed_provider = get_embedding_provider()
    for chunk in chunks:
        chunk_text = chunk["chunk_text"]
        print(f"  Generating embedding for chunk {chunk['chunk_index']} ({len(chunk_text)} chars)...")
        chunk["embedding"] = embed_provider.get_embedding(chunk_text)
        
    # 4. Save chunks to database
    database.save_document_chunks(file_name, ticker, chunks)
    print(f"  [+] Successfully indexed and saved {len(chunks)} document chunks in CockroachDB.")

