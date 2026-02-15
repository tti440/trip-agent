import os, json
import pandas as pd
from neo4j import GraphDatabase
import dotenv
from tqdm import tqdm

CSV_PATH = "reltype_to_group.csv"
PROGRESS = "reltype_apoc_progress.json"
BATCH_SIZE = 5000  # reduce to 2000 if you still see memory pressure

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
DB  = os.getenv("NEO4J_DATABASE")

CYPHER = """
CALL apoc.periodic.iterate(
  "MATCH ()-[r:REL]->() WHERE r.rel_type = $rel_type RETURN r",
  "SET r.rel_group = $rel_group,
       r.rel_score = $rel_score,
       r.rel_type_en = $rel_type_en,
       r.rel_type_norm = $rel_type_norm",
  {batchSize: $batchSize, parallel: false, params: $params}
)
YIELD batches, total, timeTaken, committedOperations, failedOperations, errorMessages
RETURN batches, total, timeTaken, committedOperations, failedOperations, errorMessages;
"""

def load_progress():
    try:
        with open(PROGRESS, "r", encoding="utf-8") as f:
            return set(json.load(f).get("done", []))
    except FileNotFoundError:
        return set()
    except Exception:
        return set()

def save_progress(done_set):
    with open(PROGRESS, "w", encoding="utf-8") as f:
        json.dump({"done": sorted(done_set)}, f, ensure_ascii=False, indent=2)

def main():
    df = pd.read_csv(CSV_PATH)
    for col in ["rel_type_en", "rel_type_norm", "rel_score"]:
        if col not in df.columns:
            df[col] = None
    df["rel_score"] = pd.to_numeric(df["rel_score"], errors="coerce")

    rows = df[["rel_type","rel_group","rel_score","rel_type_en","rel_type_norm"]].to_dict("records")

    done = load_progress()
    print(f"Resume: {len(done)} rel_types already done")

    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    with driver.session(database=DB) as session:
        # quick APOC check (will throw if not installed)
        session.run("RETURN apoc.version() AS v").single()

        for row in tqdm(rows, desc="Updating rel_type via APOC"):
            if row["rel_type"] in done:
                continue

            params = {
                "rel_type": row["rel_type"],
                "rel_group": row["rel_group"],
                "rel_score": row["rel_score"],
                "rel_type_en": row["rel_type_en"],
                "rel_type_norm": row["rel_type_norm"],
            }

            res = session.execute_write(
                lambda tx: tx.run(
                    CYPHER,
                    batchSize=BATCH_SIZE,
                    params=params,
                    **params
                ).single()
            )

            done.add(row["rel_type"])
            save_progress(done)

    driver.close()
    print("✅ Done")

if __name__ == "__main__":
    main()
