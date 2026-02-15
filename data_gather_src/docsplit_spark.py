from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim
from langchain.text_splitter import TokenTextSplitter
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document
import json

# --- Config ---
INPUT_CSV = "/app/concat_df1.csv"  # adjust name if needed
OUTPUT_DIR = "/app/document_split"  # Spark will create part files here
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50

# --- Spark session ---
spark = SparkSession.builder \
    .appName("LangChainDocSplit") \
    .master("spark://spark-master:7077") \
    .config("spark.driver.host", "travel-app") \
    .config("spark.executor.memory", "5g") \
    .config("spark.driver.memory", "5g") \
    .getOrCreate()

# --- Load CSV ---
df = (
    spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("multiLine", "true")   # critical: allow multi-line text fields
        .option("quote", '"')          # standard CSV quote
        .option("escape", '"')         # escape quotes inside text
        .csv(INPUT_CSV)
)

from pyspark.sql import functions as F

# clean up text column: normalize newlines + trim
df = (
    df.withColumn("text", F.regexp_replace(F.col("text"), r"\r\n?", "\n"))
      .withColumn("text", F.trim(F.col("text")))
      .filter(F.col("text").isNotNull() & (F.col("text") != ""))
)


# --- Convert to Document + split ---
def row_to_chunks(row):
    from langchain.text_splitter import TokenTextSplitter
    try:
        from langchain_core.documents import Document
    except ImportError:
        from langchain.schema import Document

    text = row['text']
    meta = {
        "qid": row['qid'],
        "qidLabel": row['qidLabel'],
        "website": row['website'],
        "final_url": row['final_url'],
        "lang": row['lang'],
        "doc_id": row['doc_id']
    }
    doc = Document(page_content=text, metadata=meta)
    splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return [ {"text": chunk.page_content, "metadata": chunk.metadata}
             for chunk in splitter.split_documents([doc]) ]

# --- Apply in Spark ---
rdd = df.rdd.flatMap(row_to_chunks)


import shutil, os

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

rdd.map(lambda x: json.dumps(x, ensure_ascii=False)) \
   .saveAsTextFile(OUTPUT_DIR)


print("✅ Done: documents split and saved to", OUTPUT_DIR)
