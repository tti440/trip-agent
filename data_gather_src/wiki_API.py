import re, time, math, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import tqdm

tmp_list = json.load(open('qid_landmark_set.json', 'r'))

API = "https://www.wikidata.org/w/api.php"
HEADERS = {
    "User-Agent": "travel-recs-bulk/0.1 (contact@example.com)",
}

def to_qid(x: str) -> str | None:
    if not x: return None
    x = x.strip()
    if x.startswith("wd:"): x = x[3:]
    if x.startswith("http"): x = x.rsplit("/", 1)[-1]
    return x

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_entities(ids, session, retries=4):
    """Call wbgetentities for up to 50 ids (QIDs) and return the 'entities' dict."""
    params = {
        "action": "wbgetentities",
        "ids": "|".join(ids),
        "props": "labels|descriptions|sitelinks|claims",
        "languages": "|".join(PREF_LANGS),
        "format": "json",
        "maxlag": "5",
    }
    for attempt in range(retries):
        r = session.get(API, params=params, headers=HEADERS, timeout=60)
        # Handle maxlag/rate politely
        if r.status_code in (429, 502, 503, 504):
            time.sleep(1.5 * (attempt + 1))
            continue
        r.raise_for_status()
        j = r.json()
        # If server asks to slow down due to maxlag it returns error code
        if "error" in j and j["error"].get("code") == "maxlag":
            time.sleep(2.0 * (attempt + 1))
            continue
        return j.get("entities", {})
    # last try
    r.raise_for_status()
PREF_LANGS = [
    "en","en-gb","en-ca","de","fr","es","it","pt","ru",
    "zh-hans","zh-hant","ja","ko","ar","hi"
]

def _pick_label(ent, pref_langs=PREF_LANGS, allow_desc=True, allow_sitelink=True):
    """Best-effort text for an entity: labels (with language fallback) →
       any label → description → sitelink → None."""
    labels = ent.get("labels", {}) or {}

    # pass 1: preferred languages
    for L in pref_langs:
        v = labels.get(L, {}).get("value")
        if v:
            return v

    # pass 2: any label
    for obj in labels.values():
        v = obj.get("value")
        if v:
            return v

    # pass 3: description fallback
    if allow_desc:
        descs = ent.get("descriptions", {}) or {}
        for L in pref_langs:
            v = descs.get(L, {}).get("value")
            if v:
                return v
        for obj in descs.values():
            v = obj.get("value")
            if v:
                return v

    # pass 4: sitelink title (e.g., enwiki)
    if allow_sitelink:
        sites = ent.get("sitelinks", {}) or {}
        for key in ("enwiki","enwikivoyage","enwikiquote"):
            title = sites.get(key, {}).get("title")
            if title:
                return title.replace("_", " ")
        # any sitelink
        for obj in sites.values():
            title = obj.get("title")
            if title:
                return title.replace("_", " ")

    return None

def _pick_official_site(ent):
    """Return the first URL from P856 (official website) if present."""
    claims = ent.get("claims", {}) or {}
    for snak in claims.get("P856", []) or []:
        m = snak.get("mainsnak", {})
        if m.get("snaktype") == "value":
            dv = m.get("datavalue", {})
            if dv.get("type") == "string":
                url = (dv.get("value") or "").strip()
                if url:
                    return url
    return None

def _pick_wikipedia_url(ent, pref_langs=PREF_LANGS):
    """Prefer enwiki → preferred langs’ wiki → any wiki sitelink. Returns full https URL."""
    sites = ent.get("sitelinks", {}) or {}

    # 1) enwiki
    if "enwiki" in sites and sites["enwiki"].get("title"):
        return "https://en.wikipedia.org/wiki/" + sites["enwiki"]["title"].replace(" ", "_")

    # 2) preferred language wikis (e.g., jawiki, dewiki, frwiki…)
    # Map language codes to their Wikipedia project keys (xxwiki)
    for lang in pref_langs:
        key = f"{lang.split('-')[0]}wiki"
        if key in sites and sites[key].get("title"):
            title = sites[key]["title"].replace(" ", "_")
            domain_lang = lang.split('-')[0]
            return f"https://{domain_lang}.wikipedia.org/wiki/{title}"

    # 3) any wiki
    for key, obj in sites.items():
        if key.endswith("wiki") and obj.get("title"):
            lang = key[:-4]  # strip 'wiki'
            title = obj["title"].replace(" ", "_")
            return f"https://{lang}.wikipedia.org/wiki/{title}"

    return None

def extract_fields(qid, ent):
    """Return basic fields and referenced country/location QIDs."""
    # label with fallback (never a dict)
    label = _pick_label(ent)
    # final normalization: keep string or set to ""
    if label is None:
        label = ""  # or f"{qid}" if you prefer using QID as placeholder
    claims = ent.get("claims", {})

    # P17 (country) can be multiple
    countries = []
    for snak in claims.get("P17", []) or []:
        m = snak.get("mainsnak", {})
        if m.get("snaktype") == "value" and m.get("datavalue", {}).get("type") == "wikibase-entityid":
            eid = m["datavalue"]["value"].get("id")
            if eid: countries.append(eid)

    # P131 (admin location) can be multiple
    locations = []
    for snak in claims.get("P131", []) or []:
        m = snak.get("mainsnak", {})
        if m.get("snaktype") == "value" and m.get("datavalue", {}).get("type") == "wikibase-entityid":
            eid = m["datavalue"]["value"].get("id")
            if eid: locations.append(eid)

    # P625 coordinates (use first if many)
    lat, lon = None, None
    for snak in claims.get("P625", []) or []:
        m = snak.get("mainsnak", {})
        if m.get("snaktype") == "value" and m.get("datavalue", {}).get("type") == "globecoordinate":
            v = m["datavalue"]["value"]
            lat, lon = v.get("latitude"), v.get("longitude")
            break
    
    official_site = _pick_official_site(ent)
    wikipedia_page = _pick_wikipedia_url(ent)
    site = None
    if official_site:
        site = official_site
    elif wikipedia_page:
        site = wikipedia_page
    return {
        "qid": qid,
        "qidLabel": label,
        "country": countries[0] if countries else None,
        "location": locations[0] if locations else None,
        "lat": lat,
        "lon": lon,
        "website": site
    }, set(countries) | set(locations)

def add_labels(label_map, ids, session):
    """Fetch en labels for referenced ids (countries/locations) in batches."""
    for batch in chunk(list(ids), 50):
        ents = fetch_entities(batch, session)
        for q, e in ents.items():
            if not isinstance(e, dict):  # skip missing
                continue
            lab = e.get("labels", {}).get("en", {}).get("value")
            if lab:
                label_map[q] = lab
        time.sleep(0.25)  # polite pacing

# -------------------------
# MAIN
# -------------------------
# 1) Normalize your input QIDs (take index 1 from your tmp_list items)
raw_ids = [row[1] for row in tmp_list]
qids = []
seen = set()
for x in raw_ids:
    q = to_qid(x)
    if q and q not in seen:
        seen.add(q)
        qids.append(q)

print(f"Total unique QIDs: {len(qids)}")

# 2) Parallel fetch entities in batches of 50
session = requests.Session()
rows = []
ref_ids = set()

BATCH = 50
WORKERS = 10  # keep reasonable to be polite

def work(batch):
    ents = fetch_entities(batch, session)
    local_rows = []
    local_refs = set()
    for q in batch:
        ent = ents.get(q)
        if isinstance(ent, dict):
            row, refs = extract_fields(q, ent)
            local_rows.append(row)
            local_refs |= refs
    return local_rows, local_refs

futures = []
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    for batch in chunk(qids, BATCH):
        futures.append(ex.submit(work, batch))
        time.sleep(0.05)  # tiny stagger

    for fut in as_completed(futures):
        r, refs = fut.result()
        rows.extend(r)
        ref_ids |= refs

print(f"Primary pass: {len(rows)} rows, {len(ref_ids)} referenced QIDs for labels")

# 3) Fetch labels for referenced country/location QIDs
label_map = {}
add_labels(label_map, ref_ids, session)

# 4) Attach countryLabel/locationLabel
for r in rows:
    if r["country"]:
        r["countryLabel"] = label_map.get(r["country"])
    else:
        r["countryLabel"] = None
    if r["location"]:
        r["locationLabel"] = label_map.get(r["location"])
    else:
        r["locationLabel"] = None

df = pd.DataFrame(rows, columns=[
    "qid", "qidLabel", "country", "countryLabel", "location", "locationLabel", "lat", "lon", "website"
])
df.to_csv("wikidata_landmarks_link.csv", index=False)
print("Saved", len(df), "rows to wikidata_landmarks_link.csv")
