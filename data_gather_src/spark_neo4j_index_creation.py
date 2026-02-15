import os
import socket
import requests
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import pandas_udf, col
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, StringType, DoubleType, LongType, IntegerType
import dotenv

# 1. Load Environment Variables
dotenv.load_dotenv("neo4j_acc.txt")

URI = os.getenv("NEO4J_LOCAL_URI") or "bolt://localhost:7687"
USER = os.getenv("NEO4J_LOCAL_USER") or "neo4j"
PWD = os.getenv("NEO4J_LOCAL_PASSWORD") or "password"
DATABASE = os.getenv("NEO4J_LOCAL_DATABASE") or "neo4j"
DRIVER_HOST = socket.gethostbyname(socket.gethostname())
# spark: use Scala-matching connector (very often 2.12)
spark = SparkSession.builder \
	.appName("Neo4jEmbeddingBackfill") \
    .master("spark://spark-master:7077") \
	.config("spark.driver.host", DRIVER_HOST) \
	.config("spark.jars.packages", "org.neo4j:neo4j-connector-apache-spark_2.12:5.3.2_for_spark_3") \
	.config("spark.driver.maxResultSize", "2g") \
	.config("spark.neo4j.bolt.url", URI) \
	.config("spark.neo4j.bolt.user", USER) \
	.config("spark.neo4j.bolt.password", PWD) \
	.config("spark.neo4j.database", DATABASE) \
	.config("spark.sql.execution.arrow.pyspark.enabled", "true") \
	.getOrCreate()

# @pandas_udf(ArrayType(FloatType()))
# def get_embedding(texts: pd.Series) -> pd.Series:
# 	results = []
# 	ollama_url = "http://ollama-service:11434/api/embeddings"
# 	for text in texts:
# 		if not text:
# 			results.append([])
# 			continue
# 		try:
# 			r = requests.post(ollama_url, json={"model": "nomic-embed-text", "prompt": text, "keep_alive": 0, "options": {"num_ctx": 2048}})
# 			results.append(r.json().get("embedding", []) if r.status_code == 200 else [])
# 		except Exception:
# 			results.append([])
# 	return pd.Series(results)

_SESSION = None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
    return _SESSION

@pandas_udf(ArrayType(FloatType()))
def get_embedding(texts: pd.Series) -> pd.Series:
    session = _get_session()
    ollama_url = "http://ollama-service:11434/api/embed"
    
    input_texts = texts.tolist()
    out = [[] for _ in input_texts]
    idx = [i for i, t in enumerate(input_texts) if t]
    payload_texts = [input_texts[i] for i in idx]
    
    batch_size = 32
    for start in range(0, len(payload_texts), batch_size):
        batch = payload_texts[start:start + batch_size]
        batch_idx = idx[start:start + batch_size]

        try:
            r = session.post(
                ollama_url,
                json={
                    "model": "nomic-embed-text",
                    "input": batch,
                    # Good defaults:
                    "truncate": True,          # or False if you prefer hard errors
                    "keep_alive": "10m",       # or -1 to keep loaded indefinitely
                    "options": {"num_ctx": 2048},  # only if you have a reason
                },
            )

            if r.status_code == 200:
                embs = r.json().get("embeddings", [])
                for j, vec in enumerate(embs[:len(batch_idx)]):
                    out[batch_idx[j]] = vec
            # else: leave as [] (matches old behavior)
        except Exception:
            # leave as [] (matches old behavior)
            pass

    return pd.Series(out)


schema = StructType([
	StructField("id", StringType(), True),
	StructField("text", StringType(), True),
])

read_query = """
MATCH (d:Document)
WHERE d.text IS NOT NULL
RETURN toString(d.id) AS id, d.text AS text
"""

df = spark.read.format("org.neo4j.spark.DataSource") \
	.schema(schema) \
	.option("url", URI) \
	.option("authentication.type", "basic") \
	.option("authentication.basic.username", USER) \
	.option("authentication.basic.password", PWD) \
	.option("database", DATABASE) \
	.option("query", read_query) \
	.load()

if df.rdd.isEmpty():
	print("No docs missing embeddings.")
else:
	df_with_embeddings = df.repartition(10).withColumn("embedding", get_embedding(col("text")))

	write_query = """
	MATCH (d:Document {id: event.id})
	SET d.embedding = event.embedding
	"""

	df_with_embeddings.write.format("org.neo4j.spark.DataSource") \
	.mode("Append") \
	.option("url", URI) \
	.option("authentication.type", "basic") \
	.option("authentication.basic.username", USER) \
	.option("authentication.basic.password", PWD) \
	.option("database", DATABASE) \
	.option("query", write_query) \
	.option("batch.size", "1000") \
	.option("parallelism", "10") \
    .option("retry.attempts", "10") \
	.save()