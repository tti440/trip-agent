# -*- coding: utf-8 -*-
"""
High-throughput, stall-proof crawler:
- ThreadPoolExecutor for network fetches
- ProcessPoolExecutor for parsing (trafilatura/pdf/langdetect)
- Never blocks the main loop on a single slow parse
- Retries + timeouts + per-domain throttle
- UTF-8 logging; progress to console, details to file
- Periodic parquet checkpoints, resumable
"""

import os, sys, time, re, json, hashlib, mimetypes, threading, warnings
from io import BytesIO
from urllib.parse import urlparse
from urllib import robotparser

import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import tldextract
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from charset_normalizer import from_bytes
from langdetect import detect, LangDetectException
from pypdf import PdfReader
import trafilatura

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from tqdm import tqdm
import logging


# ----------------------- Config -----------------------

UA = "travel-recs-crawler/0.1 (+contact@example.com)"
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 15
TIMEOUT = (CONNECT_TIMEOUT, READ_TIMEOUT)        # (connect, read)
RETRIES = 2                                      # per request
BACKOFF = 0.3
PER_DOMAIN_DELAY = 0.01                          # polite throttle
SLEEP_BETWEEN_REQUESTS = 0

SAVE_EVERY = 50000                             # checkpoint rows
MAX_FETCH_WORKERS = 12                          # threads
MAX_PARSE_WORKERS = 12                            # processes
PARSE_TIMEOUT = 30                               # seconds per page extraction
TIME_BUDGET = 1.5 * 3600                         # seconds
PART_DIR = "parts"
os.makedirs(PART_DIR, exist_ok=True)

LOG_FILE = "crawler_run.log"

# Make Windows console UTF-8 (avoids stalls on accents)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Logging: progress to console, full details to file
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logger = logging.getLogger("crawler")
logger.setLevel(logging.DEBUG)

# File (everything)
fh = FlushFileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(fh)

# Console (progress only)
# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.INFO)
# ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# logger.addHandler(ch)


# ----------------------- HTTP session -----------------------

_session = requests.Session()
_session.headers.update({
    "User-Agent": UA,
    "Accept": "text/html,application/pdf,application/xml,text/xml;q=0.9,*/*;q=0.8",
})

# large connection pools so threads don't serialize
adapter = HTTPAdapter(
    pool_connections=MAX_FETCH_WORKERS * 2,
    pool_maxsize=MAX_FETCH_WORKERS * 4,
    max_retries=Retry(
        total=RETRIES,
        backoff_factor=BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    ),
)
_session.mount("http://", adapter)
_session.mount("https://", adapter)


# ----------------------- Throttle & robots -----------------------

_last_hit = {}
_last_hit_lock = threading.Lock()
parse_sema = threading.Semaphore(MAX_PARSE_WORKERS * 2)

def throttle(url: str):
    d = tldextract.extract(url)
    domain = f"{d.domain}.{d.suffix}"
    now = time.time()
    with _last_hit_lock:
        last = _last_hit.get(domain, 0.0)
        wait = max(0.0, PER_DOMAIN_DELAY - (now - last))
        if wait:
            time.sleep(wait)
        _last_hit[domain] = time.time()

_robots_cache = {}
_robots_lock = threading.Lock()

def load_robots(base_url: str):
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    with _robots_lock:
        rp = _robots_cache.get(robots_url)
        if rp:
            return rp
    try:
        throttle(robots_url)
        resp = _session.get(robots_url, timeout=TIMEOUT)
        rp = robotparser.RobotFileParser()
        rp.parse(resp.text.splitlines())
        with _robots_lock:
            _robots_cache[robots_url] = rp
        return rp
    except Exception:
        return None

def allowed_by_robots(url: str) -> bool:
    rp = load_robots(url)
    if not rp:
        return True
    return rp.can_fetch(UA, url)


# ----------------------- Fetch & parse -----------------------

def fetch(url: str):
    throttle(url)
    if SLEEP_BETWEEN_REQUESTS:
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    _start = time.time()
    r = _session.get(url, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    ct = (r.headers.get("Content-Type") or "").lower()
    elapsed = time.time() - _start
    # Detailed per-URL logs go to file; console shows only progress counters
    logger.debug(f"Fetched {url} in {elapsed:.2f}s (ct={ct})")
    return r.url, ct, r.content


# Silence bs4 XML-as-HTML warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def extract_html(html_bytes: bytes, base_url: str) -> str:
    # Quick XML sniff
    head = html_bytes.lstrip()[:400].lower()
    looks_xml = (
        head.startswith(b"<?xml")
        or b"<rss" in head
        or b"<feed" in head
        or b"<urlset" in head
        or b"<sitemapindex" in head
    )

    # Normalize encoding once
    try:
        norm_text = str(from_bytes(html_bytes).best())
    except Exception:
        norm_text = html_bytes.decode("utf-8", "replace")
    norm_bytes = norm_text.encode("utf-8")

    # Try trafilatura (JSON mode)
    try:
        data = trafilatura.extract(
            norm_bytes,
            url=base_url,
            output_format="json",
            favor_recall=False,
            include_comments=False,
            include_links=False,
        )
        if data:
            obj = json.loads(data)
            title = (obj.get("title") or "").strip()
            text = (obj.get("text") or "").strip()
            combined = (f"{title}\n\n{text}".strip() if title else text)
            if combined and len(combined.split()) > 60:
                return combined
    except Exception:
        pass

    # XML branch
    if looks_xml:
        soup = BeautifulSoup(norm_text, "xml")
        texts = []
        for el in soup.find_all(["title"]):
            t = el.get_text(" ", strip=True)
            if t and len(t) > 5:
                texts.append(t)
        for el in soup.find_all(["description", "summary"]):
            t = el.get_text(" ", strip=True)
            if t and len(t) > 40:
                texts.append(t)
        out = "\n\n".join(dict.fromkeys(texts))
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out

    # HTML fallback
    try:
        soup = BeautifulSoup(norm_text, "lxml")
    except Exception:
        soup = BeautifulSoup(norm_text, "html5lib")

    for tag in soup(["script","style","noscript","header","footer","nav","form","iframe","svg"]):
        tag.decompose()

    texts = []
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    if title: texts.append(title)
    for h in soup.find_all(["h1","h2","h3"]):
        txt = h.get_text(" ", strip=True)
        if txt and len(txt) > 5:
            texts.append(txt)
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if txt and len(txt) > 40:
            texts.append(txt)

    out = "\n\n".join(texts)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()

def extract_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            txt = re.sub(r"[ \t]+", " ", txt)
            pages.append(txt.strip())
        text = "\n\n".join(p for p in pages if p)
        return text.strip()
    except Exception:
        return ""

def clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(cookie|consent|analytics).{0,40}", "", text, flags=re.I)
    return text.strip()

def detect_lang(text: str):
    try:
        return detect(text)
    except LangDetectException:
        return None


# ----------------------- Worker (threaded) -----------------------

def harvest_one(row: pd.Series) -> dict:
    """Fetch + parse in the same thread (no IPC)."""
    url = row["website"]

    if not allowed_by_robots(url):
        return dict(row, status="fail", reason="robots_disallow")

    try:
        final_url, ct, content = fetch(url)
    except Exception as e:
        return dict(row, status="fail", reason=f"fetch_error:{e}")

    text = ""
    if content[:4] == b"%PDF" or "pdf" in final_url.lower():
        text = extract_pdf(content)
    else:
        if b"<" in content[:1000]:
            text = extract_html(content, final_url)

    text = clean_text(text or "")
    if not text or len(text.split()) < 5:
        return dict(row, status="fail", reason="too_short_or_empty")

    lang = detect_lang(text)
    doc_id = hashlib.md5(final_url.encode("utf-8")).hexdigest()
    return dict(
        qid=row["qid"],
        qidLabel=row["qidLabel"],
        website=row["website"],
        final_url=final_url,
        status="ok",
        doc_id=doc_id,
        lang=lang,
        text=text,
    )

# ----------------------- Main runner -----------------------

def run_batch_resumable(input_csv, limit=None):
    df = pd.read_csv(input_csv)
    df = df[df["website"].notna() & (df["website"] != "")]

    if limit:
        df = df.head(limit)

    # --- resumption: skip already processed ---
    processed_ids = set()
    if os.path.exists("concat_df.csv"):
        part_df = pd.read_csv("concat_df.csv", usecols=["qid"])
        processed_ids.update(part_df["qid"].dropna().unique())
    for file_name in os.listdir(PART_DIR):
        if file_name.endswith(".parquet"):
            part_df = pd.read_parquet(f"parts/{file_name}", columns=["qid"])
            processed_ids.update(part_df["qid"].dropna().unique())
    print(f"Already processed: {len(processed_ids)} ids")

    df = df[~df["qid"].isin(processed_ids)]
    print(f"To process: {df.shape[0]} rows")

    # --- resume part numbering ---
    existing_parts = [f for f in os.listdir(PART_DIR)
                      if f.startswith("part_") and f.endswith(".parquet")]
    if existing_parts:
        last_part_num = max(int(f[5:10]) for f in existing_parts)
    else:
        last_part_num = -1
    next_part_index = last_part_num + 1

    start = time.time()
    buffer = []

    with ThreadPoolExecutor(max_workers=MAX_FETCH_WORKERS) as pool:
        futures = {pool.submit(harvest_one, row): row for _, row in df.iterrows()}
        logger.info("Crawler started")

        done_count = 0
        total_count = len(futures)

        for fut in as_completed(futures):
            row = futures[fut]
            try:
                result = fut.result(timeout=PARSE_TIMEOUT)
                buffer.append(result)
            except Exception as e:
                buffer.append(dict(row, status="fail", reason=f"executor_error:{e}"))

            done_count += 1
            if done_count % 100 == 0 or done_count == total_count:
                logger.info(f"[{done_count}/{total_count}] processed")

            # checkpoint
            if len(buffer) >= SAVE_EVERY:
                part_file = os.path.join(PART_DIR, f"part_{next_part_index:05d}.parquet")
                pq.write_table(pa.Table.from_pandas(pd.DataFrame(buffer)),
                               part_file, compression="snappy")
                logger.info(f"Wrote {len(buffer)} rows → {part_file}")
                print(f"Wrote {len(buffer)} rows → {part_file}")
                buffer.clear()
                next_part_index += 1

    # flush remaining
    if buffer:
        part_file = os.path.join(PART_DIR, f"part_{next_part_index:05d}.parquet")
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(buffer)),
                       part_file, compression="snappy")
        print(f"Wrote {len(buffer)} rows → {part_file}")


# ----------------------- CLI -----------------------

if __name__ == "__main__":
    run_batch_resumable("wikidata_landmarks_link_enriched.csv", limit=None)
