#!/usr/bin/env python3
import os
import glob
import pickle
import argparse
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm
from neo4j import GraphDatabase
import dotenv
from langchain_neo4j import Neo4jGraph


ATTACH_METADATA_CYPHER = """
UNWIND $rows AS row
MATCH (d:Document {qid: row.qid, source_tag: row.source_tag})

FOREACH (_ IN CASE WHEN row.country IS NULL THEN [] ELSE [1] END |
  MERGE (c:Country {id: row.country})
  MERGE (d)-[:LOCATED_IN_COUNTRY]->(c)
)

FOREACH (_ IN CASE WHEN row.location IS NULL THEN [] ELSE [1] END |
  MERGE (l:Location {id: row.location})
  MERGE (d)-[:LOCATED_IN]->(l)
)

FOREACH (_ IN CASE WHEN row.coordinates IS NULL THEN [] ELSE [1] END |
  MERGE (co:Coordinates {lat: row.coordinates.lat, lon: row.coordinates.lon})
  MERGE (d)-[:HAS_GEOGRAPHIC_LOCATION]->(co)
)
"""


def load_env(path: str):
	if not dotenv.load_dotenv(path):
		raise RuntimeError(f"Failed to load env file: {path}")


def get_creds() -> Tuple[str, str, str]:
	uri = os.getenv("NEO4J_LOCAL_URI")
	user = os.getenv("NEO4J_LOCAL_USER")
	pwd = os.getenv("NEO4J_LOCAL_PASSWORD")
	db = os.getenv("NEO4J_DATABASE") or "neo4j"

	if not uri or not user or not pwd:
		raise RuntimeError("Missing NEO4J_LOCAL_URI / USER / PASSWORD in env file.")
	return uri, user, pwd, db


def verify_connectivity(uri: str, user: str, pwd: str):
	with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
		driver.verify_connectivity()


def find_pkl_files(graph_dirs: List[str]) -> List[str]:
	"""
	IMPORTANT: Do NOT deduplicate. We keep full paths even if filenames collide.
	"""
	files: List[str] = []
	for d in graph_dirs:
		if not os.path.isdir(d):
			print(f"⚠️  Skipping missing directory: {d}")
			continue
		files.extend(glob.glob(os.path.join(d, "**", "graph_docs_batch_*.pkl"), recursive=True))

	if not files:
		raise FileNotFoundError("No graph_docs_batch_*.pkl files found in given directories.")

	# Deterministic ordering: first by parent dir, then by filename
	files.sort(key=lambda p: (os.path.dirname(p), os.path.basename(p)))
	return files


def safe_get_qid(graph_doc: Any) -> Optional[str]:
	src = getattr(graph_doc, "source", None)
	meta = getattr(src, "metadata", None) if src else None
	if isinstance(meta, dict):
		qid = meta.get("qid")
		return str(qid) if qid else None
	return None

def set_source_tag(graph_docs, source_tag: str):
	for gd in graph_docs:
		src = getattr(gd, "source", None)
		meta = getattr(src, "metadata", None) if src else None
		if isinstance(meta, dict):
			meta["source_tag"] = source_tag
   
def sanitize_text(text):
    if isinstance(text, str):
        # Aggressively strip lone surrogates (the cause of the crash)
        return text.encode('utf-8', 'ignore').decode('utf-8')
    return text

def deep_clean_graph_doc(doc):
    """Recursively cleans nodes, relationships, and source metadata."""
    
    # 1. Clean Source Document
    if hasattr(doc, 'source') and doc.source:
        if hasattr(doc.source, 'page_content'):
            doc.source.page_content = sanitize_text(doc.source.page_content)
        if hasattr(doc.source, 'metadata') and isinstance(doc.source.metadata, dict):
            doc.source.metadata = {k: sanitize_text(v) for k, v in doc.source.metadata.items()}

    # 2. Clean Nodes (ID, Type, Properties)
    if hasattr(doc, 'nodes'):
        for node in doc.nodes:
            node.id = sanitize_text(node.id)
            node.type = sanitize_text(node.type)
            if hasattr(node, 'properties') and isinstance(node.properties, dict):
                node.properties = {k: sanitize_text(v) for k, v in node.properties.items()}

    # 3. Clean Relationships (Type, Properties, Source/Target IDs)
    if hasattr(doc, 'relationships'):
        for rel in doc.relationships:
            rel.type = sanitize_text(rel.type)
            if hasattr(rel, 'properties') and isinstance(rel.properties, dict):
                rel.properties = {k: sanitize_text(v) for k, v in rel.properties.items()}
            
            # Ensure the connected node refs are also clean
            if hasattr(rel, 'source') and hasattr(rel.source, 'id'):
                rel.source.id = sanitize_text(rel.source.id)
            if hasattr(rel, 'target') and hasattr(rel.target, 'id'):
                rel.target.id = sanitize_text(rel.target.id)

    return doc

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--dotenv", default="neo4j_acc.txt")
	ap.add_argument("--graph_dirs", nargs="+", required=True)
	ap.add_argument("--csv", default=None)
	ap.add_argument("--country_col", default="countryLabel")
	ap.add_argument("--location_col", default="locationLabel")
	ap.add_argument("--lat_col", default="lat")
	ap.add_argument("--lon_col", default="lon")
	ap.add_argument("--skip_metadata", action="store_true")
	args = ap.parse_args()

	load_env(args.dotenv)
	uri, user, pwd, db = get_creds()
	print(f"Using Neo4j URI: {uri}")
	verify_connectivity(uri, user, pwd)
	print("✅ Neo4j connection OK")

	pkl_files = find_pkl_files(args.graph_dirs)
	print(f"Found {len(pkl_files)} PKL files (including same-name files in different dirs).")

	df = None
	if args.csv and not args.skip_metadata:
		df = pd.read_csv(args.csv)
		if "qid" not in df.columns:
			raise ValueError("CSV must contain a 'qid' column.")

	graph = Neo4jGraph(url=uri, username=user, password=pwd, database=db)

	# We still de-dup *metadata attachment* by qid so we don’t spam edges like LOCATED_IN_COUNTRY
	seen_keys: Set[Tuple[str, str]] = set()
	rows: List[Dict[str, Any]] = []

	for pkl_path in tqdm(pkl_files, desc="Ingesting PKLs"):
		with open(pkl_path, "rb") as f:
			graph_docs = pickle.load(f)
		graph_docs = [deep_clean_graph_doc(d) for d in graph_docs]
		# Ingest everything from this file
		source_tag = os.path.basename(os.path.dirname(pkl_path))
		set_source_tag(graph_docs, source_tag)
		graph.add_graph_documents(
			graph_docs,
			baseEntityLabel=True,
			include_source=True,
		)

		if df is not None:
			batch_qids = {safe_get_qid(gd) for gd in graph_docs}
			batch_qids = {q for q in batch_qids if q}
			batch_qids = {safe_get_qid(gd) for gd in graph_docs}
			batch_qids = {q for q in batch_qids if q}
			new_pairs = {(qid, source_tag) for qid in batch_qids} - seen_keys

			for (qid, source_tag) in new_pairs:
				r = df[df["qid"] == qid]
				if r.empty:
					continue
				r = r.iloc[0]
				rows.append({
					"qid": qid,
					"source_tag": source_tag,
					"country": r[args.country_col] if args.country_col in r and pd.notna(r[args.country_col]) else None,
					"location": r[args.location_col] if args.location_col in r and pd.notna(r[args.location_col]) else None,
					"coordinates": (
						{"lat": float(r[args.lat_col]), "lon": float(r[args.lon_col])}
						if args.lat_col in r and args.lon_col in r
						and pd.notna(r[args.lat_col]) and pd.notna(r[args.lon_col])
						else None
					)
				})

			seen_keys |= new_pairs

	if df is not None and rows:
		print(f"Attaching metadata for {len(rows)} qids...")
		with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
			with driver.session(database=db) as session:
				session.execute_write(lambda tx: tx.run(ATTACH_METADATA_CYPHER, rows=rows))
		print("✅ Metadata attached.")

	print("✅ Done.")


if __name__ == "__main__":
	main()
