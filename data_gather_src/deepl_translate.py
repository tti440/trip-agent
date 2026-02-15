#!/usr/bin/env python3
import os, time, random, re, json
import dotenv
import pandas as pd
import deepl
from tqdm import tqdm

INPUT = "rel_type.csv"
COL = "r.rel_type"
OUT_CSV = "reltype_translation_map_deepl.csv"
CACHE_JSON = "reltype_translation_cache_deepl.json"
TARGET_LANG = "EN-US"
BATCH_SIZE = 50
MAX_RETRIES = 8

def normalize_predicate_en(s: str) -> str:
    s = (s or "").strip().strip("\"'`")
    s = s.replace("-", "_").replace(" ", "_")
    s = re.sub(r"[^\w_]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()

def load_cache(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def save_cache(path: str, cache: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def is_overload_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("too many requests" in msg) or ("high load" in msg) or ("429" in msg)

def translate_batch(translator: deepl.DeepLClient, batch: list[str]) -> list[str]:
    # DeepL accepts a list and returns list-like results
    results = translator.translate_text(batch, target_lang=TARGET_LANG)
    out = []
    for res in results:
        text = res.text if hasattr(res, "text") else str(res)
        out.append(text)
    return out

def main():
    dotenv.load_dotenv()
    api_key = os.getenv("DEEPL_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPL_API_KEY missing. Put it in .env or export it.")
    translator = deepl.DeepLClient(api_key)

    df = pd.read_csv(INPUT)
    if COL not in df.columns:
        raise RuntimeError(f"Column '{COL}' not found. Found: {list(df.columns)}")

    rels = df[COL].dropna().astype(str).str.strip()
    rels = rels[rels != ""]
    rels_unique = rels.drop_duplicates().tolist()
    print(f"Distinct predicates: {len(rels_unique)}")

    cache = load_cache(CACHE_JSON)
    print(f"Loaded cache: {len(cache)}")

    to_translate = [r for r in rels_unique if r not in cache]
    print(f"Need translation: {len(to_translate)}")

    pbar = tqdm(total=len(to_translate), desc="DeepL translating")
    i = 0
    while i < len(to_translate):
        batch = to_translate[i:i+BATCH_SIZE]

        # Retry w/ exponential backoff on overload
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                translated = translate_batch(translator, batch)
                for src, en_raw in zip(batch, translated):
                    en = normalize_predicate_en(en_raw)
                    cache[src] = en if en else normalize_predicate_en(src)
                save_cache(CACHE_JSON, cache)
                pbar.update(len(batch))
                i += len(batch)
                break
            except Exception as e:
                if attempt == MAX_RETRIES or not is_overload_error(e):
                    raise
                sleep_s = min(60.0, (2 ** (attempt - 1)) + random.random() * 0.5)
                print(f"⚠️ DeepL overloaded (attempt {attempt}/{MAX_RETRIES}). Sleeping {sleep_s:.1f}s")
                time.sleep(sleep_s)

    pbar.close()

    out_df = pd.DataFrame(
        [{"rel_type": r, "rel_type_en": cache.get(r, normalize_predicate_en(r))}
         for r in rels_unique]
    )
    out_df.to_csv(OUT_CSV, index=False)
    print(f"✅ Wrote {OUT_CSV} and cache {CACHE_JSON}")

if __name__ == "__main__":
    main()
