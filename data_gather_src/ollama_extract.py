# pip install langchain langchain-community langchain-experimental neo4j pandas langchain_ollama pyarrow

import os, math, time, json, pickle, gc
import pandas as pd
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# Adjust these based on your Ollama speed
MODEL_NAME     = os.getenv("MODEL_NAME", "llama3.1")
NUM_CTX        = int(os.getenv("NUM_CTX", "4096"))
MAX_WORKERS    = int(os.getenv("MAX_WORKERS", "12")) # Increase if Ollama is on a fast GPU
RETRIES        = int(os.getenv("RETRIES", "3"))
BACKOFF        = float(os.getenv("BACKOFF", "1.5"))
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "500"))
INNER_CHUNK    = int(os.getenv("INNER_CHUNK", "32"))
OUT_DIR        = "."
CSV_PATH       = os.getenv("CSV_PATH", "landmarks_ready_for_ollama") # Points to Spark Output



from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_ollama import ChatOllama

text_splitter = TokenTextSplitter(chunk_size=750, chunk_overlap=50)
_llm = None
_transformer = None
_lock = threading.Lock()

def make_llm():
    global _llm, _transformer
    with _lock:
        if _llm is None:
            # format="json" enforces structured output which helps with graph extraction
            _llm = ChatOllama(model=MODEL_NAME, temperature=0, format="json", num_ctx=NUM_CTX)
            _transformer = LLMGraphTransformer(llm=_llm)
    return _transformer

def convert_chunk(docs):
    transformer = make_llm()
    for attempt in range(1, RETRIES + 1):
        try:
            return transformer.convert_to_graph_documents(docs)
        except Exception:
            if attempt == RETRIES: raise
            time.sleep(BACKOFF * attempt)

def parallel_convert(batch_docs: List[Document]) -> List:
    if not batch_docs: return []
    chunks = [batch_docs[i:i+INNER_CHUNK] for i in range(0, len(batch_docs), INNER_CHUNK)]
    out = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(convert_chunk, c) for c in chunks]
        for f in as_completed(futs):
            out.extend(f.result())
    return out

def df_to_documents(df: pd.DataFrame) -> List[Document]:
    docs = []
    # Fill NaNs to avoid errors accessing attributes
    df = df.fillna("")
    
    # Use itertuples for speed
    for row in df.itertuples(index=False):
        # Support both CSV (column access) and Parquet (attribute access)
        # We try getattr first (Parquet), then fallback to dictionary access if needed
        try:
            text = str(getattr(row, "text", "")).strip()
        except AttributeError:
            continue # Skip malformed rows

        if not text: continue
        
        # Preserve metadata for the Graph Ingest step later
        meta = {
            "qid": getattr(row, "qid", None),
            "qidLabel": getattr(row, "qidLabel", None),
            "website": getattr(row, "website", None),
            "doc_id": getattr(row, "doc_id", "unknown"),
        }
        
        # Split long text immediately so we don't overflow context window
        raw_doc = Document(page_content=text, metadata=meta)
        docs.extend(text_splitter.split_documents([raw_doc]))
    return docs

def already_done_batches(out_dir):
    """Scans output directory to resume from last batch"""
    done = set()
    processed_qid = set()
    for f in os.listdir(out_dir):
        if f.startswith("graph_docs_batch_") and f.endswith(".pkl"):
            try:
                # Extract batch ID from filename
                done.add(int(f.split("_")[-1].split(".")[0]))
                
                # Extract processed QIDs to avoid re-doing work
                with open(os.path.join(out_dir, f), "rb") as pf:
                    batch = pickle.load(pf)
                    for d in batch:
                        if hasattr(d, 'source') and d.source.metadata.get("qid"):
                            processed_qid.add(d.source.metadata.get("qid"))
            except: pass
    return (max(done) if done else 0), processed_qid

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"📂 Loading data from: {CSV_PATH}")

    # --- AUTO-DETECT FORMAT (Parquet vs CSV) ---
    if os.path.isdir(CSV_PATH):
        print("   -> Detected Directory (Spark/Parquet)")
        try:
            # Pandas can read a folder of parquet files as one dataframe
            df = pd.read_parquet(CSV_PATH)
            print(f"   -> Loaded {len(df)} rows.")
            documents = df_to_documents(df)
        except Exception as e:
            print(f"   -> Error reading parquet: {e}")
            return
    elif CSV_PATH.endswith(".csv"):
        print("   -> Detected CSV File")
        documents = df_to_documents(pd.read_csv(CSV_PATH))
    else:
        print("❌ Unknown input format. Expected .csv file or Parquet directory.")
        return

    # --- Resume Logic ---
    max_num, processed_qid = already_done_batches(OUT_DIR)
    if processed_qid:
        documents = [d for d in documents if d.metadata.get("qid") not in processed_qid]
        print(f"🔄 Resuming: Skipping {len(processed_qid)} existing QIDs. {len(documents)} remain.")

    if not documents:
        print("🎉 No new documents to process. Done.")
        return

    # --- Processing Loop ---
    total = len(documents)
    num_batches = math.ceil(total / BATCH_SIZE)
    start_batch = max_num + 1
    
    print(f"🚀 Starting extraction at batch {start_batch}. Total batches: {num_batches}")
    
    for b in range(num_batches):
        start = b * BATCH_SIZE
        end = min((b + 1) * BATCH_SIZE, total)
        batch_docs = documents[start:end]
        
        print(f"   Processing batch {b+1}/{num_batches} ({len(batch_docs)} docs)...")
        
        # This is where the heavy lifting (Ollama) happens
        graph_docs = parallel_convert(batch_docs)
        
        out_path = os.path.join(OUT_DIR, f"graph_docs_batch_{start_batch + b}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(graph_docs, f)
        
        print(f"   💾 Saved -> {out_path}")
        
        # Cleanup memory
        del graph_docs, batch_docs
        gc.collect()

    print("✅ All Extractions Complete.")

if __name__ == "__main__":
    main()