import argparse
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# 1) Define your target ontology (edit freely)
GROUPS = {

    # ─────────────────────────────
    # Spatial / Geographic
    # ─────────────────────────────
    "LOCATED_IN": [
        "located in", "location of", "situated in",
        "administrative division of", "administrative region of",
        "country of", "city of", "region of", "province of",
        "municipality of", "district of", "state of",
        "belongs to country", "in country", "in city", "in region"
    ],

    "NEAR": [
        "near", "adjacent to", "next to", "close to",
        "neighbor of", "nearby", "around", "in vicinity of",
        "within walking distance", "around the corner"
    ],

    "PART_OF": [
        "part of", "component of", "within", "contained in",
        "belongs to", "included in", "subdivision of",
        "section of", "member of", "element of"
    ],

    "HAS_COORDINATES": [
        "has coordinates", "latitude", "longitude",
        "geographic coordinates", "geo location", "gps location"
    ],

    # ─────────────────────────────
    # Identity / Naming
    # ─────────────────────────────
    "ALIAS_OF": [
        "alias", "alias of", "also known as",
        "alternate name", "alternative name",
        "former name", "short name", "nickname",
        "local name", "native name"
    ],

    "IDENTIFIED_AS": [
        "identified as", "classified as", "defined as",
        "considered as", "recognized as"
    ],

    # ─────────────────────────────
    # Description / Semantics
    # ─────────────────────────────
    "ABOUT": [
        "about", "describes", "description of",
        "refers to", "mentions", "discusses",
        "related topic", "concerns", "focuses on"
    ],

    "RELATED_TO": [
        "related to", "associated with", "affiliated with",
        "aligned with", "connected to", "linked to",
        "correlated with", "interaction with"
    ],

    # ─────────────────────────────
    # Creation / Origin / History
    # ─────────────────────────────
    "CREATED_BY": [
        "created by", "designed by", "built by",
        "constructed by", "developed by", "engineered by",
        "produced by", "authored by"
    ],

    "FOUNDED_IN": [
        "founded in", "established in", "originated in",
        "formed in", "created in", "opened in"
    ],

    "DEDICATED_TO": [
        "dedicated to", "named after", "commemorates",
        "in honor of"
    ],

    # ─────────────────────────────
    # Usage / Function
    # ─────────────────────────────
    "USED_FOR": [
        "used for", "purpose", "intended for",
        "serves as", "function of", "application of"
    ],

    "ACCESSIBLE_VIA": [
        "accessed via", "accessed by", "reachable via",
        "access through", "connected via", "transport via"
    ],

    "OPERATED_BY": [
        "operated by", "managed by", "run by",
        "administered by", "maintained by"
    ],

    # ─────────────────────────────
    # Temporal
    # ─────────────────────────────
    "OCCURRED_IN": [
        "occurred in", "happened in", "took place in",
        "held in", "event in"
    ],

    "DATE_OF": [
        "date of", "year of", "time of",
        "period of", "during"
    ],

    # ─────────────────────────────
    # Cultural / Social
    # ─────────────────────────────
    "CULTURAL_ASSOCIATION": [
        "cultural association", "heritage of",
        "tradition of", "symbol of", "icon of"
    ],

    "LANGUAGE_ASSOCIATION": [
        "language of", "spoken in", "written in",
        "associated language"
    ],

    # ─────────────────────────────
    # Legal / Administrative
    # ─────────────────────────────
    "ADMINISTERED_BY": [
        "administered by", "governed by",
        "regulated by", "jurisdiction of"
    ],

    "DESIGNATED_AS": [
        "designated as", "classified as",
        "listed as", "recognized as",
        "declared as"
    ],

    # ─────────────────────────────
    # Media / Documents / Sources
    # ─────────────────────────────
    "SOURCE_OF": [
        "source of", "derived from", "origin of"
    ],

    "MENTIONED_IN": [
        "mentioned in", "appears in",
        "referenced in", "cited in"
    ],

    # ─────────────────────────────
    # Catch-all (important)
    # ─────────────────────────────
    "OTHER": [
        "other", "miscellaneous", "unspecified"
    ]
}

def normalize_text(s: str) -> str:
    s = str(s)
    s = s.replace("-", " ").replace("_", " ")
    s = " ".join(s.split())
    return s.lower().strip()


def build_group_prototypes():
    group_names = []
    group_texts = []
    for g, examples in GROUPS.items():
        # Prototype is: group name + examples (gives better anchor)
        proto = g.replace("_", " ").lower() + ": " + "; ".join(examples)
        group_names.append(g)
        group_texts.append(proto)
    return group_names, group_texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", default="reltype_deepl_translated.csv", help="CSV with distinct rel_type column")
    ap.add_argument("--col", default="rel_type_en", help="column name containing predicate strings")
    ap.add_argument("--out_csv", default="reltype_to_group.csv")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--threshold", type=float, default=0.35, help="below this -> OTHER")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.col not in df.columns:
        raise ValueError(f"Column '{args.col}' not found. Columns: {list(df.columns)}")
    rel_types = df["rel_type"].dropna().astype(str).tolist()
    rel_types_en = df[args.col].dropna().astype(str).tolist()
    rel_norm = [normalize_text(x) for x in rel_types_en]

    print(f"Loaded {len(rel_types_en)} rel_types")

    # Embedding model
    model = SentenceTransformer(args.model)

    # Encode predicates
    rel_emb = model.encode(rel_norm, batch_size=256, show_progress_bar=True, normalize_embeddings=True)

    # Encode group prototypes
    group_names, group_texts = build_group_prototypes()
    grp_emb = model.encode(group_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

    # Cosine similarity (fast with normalized embeddings: dot product)
    sims = rel_emb @ grp_emb.T  # shape: (N, G)

    best_idx = sims.argmax(axis=1)
    best_score = sims.max(axis=1)

    best_group = [group_names[i] for i in best_idx]
    # thresholding
    best_group = [g if s >= args.threshold else "OTHER" for g, s in zip(best_group, best_score)]

    out = pd.DataFrame({
        "rel_type": rel_types,
        "rel_type_en": rel_types_en,
        "rel_type_norm": rel_norm,
        "rel_group": best_group,
        "rel_score": best_score
    }).sort_values(["rel_group", "rel_score"], ascending=[True, False])

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")

    # Helpful: show distribution
    print(out["rel_group"].value_counts().head(30))


if __name__ == "__main__":
    main()
