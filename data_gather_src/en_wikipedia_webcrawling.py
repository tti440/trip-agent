import os
import pandas as pd
import requests
import trafilatura
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, when, lit, sha2

os.makedirs("landmarks_ready_for_ollama", exist_ok=True)
# Initialize Spark (Safe Mode: 4 Cores to prevent crashes)
spark = SparkSession.builder \
	.appName("LandmarkContentFetcher") \
	.master("spark://spark-master:7077") \
	.config("spark.driver.host", "travel-app") \
	.config("spark.sql.execution.arrow.pyspark.enabled", "true") \
	.getOrCreate()

# --- UDF 1: Wikipedia API Fetcher (FIXED with User-Agent) ---
@pandas_udf("string")
def fetch_wikipedia_summary(urls: pd.Series) -> pd.Series:
	results = []
	# Create a session to reuse TCP connections (faster)
	session = requests.Session()
	# CRITICAL FIX: Wikipedia requires a User-Agent header
	headers = {
		"User-Agent": "LandmarkResearchBot/1.0 (educational project; python-requests)"
	}
	
	for url in urls:
		if not url or not isinstance(url, str):
			results.append(None)
			continue
		try:
			# Convert https://en.wikipedia.org/wiki/Big_Ben -> Big_Ben
			title = url.split("/wiki/")[-1]
			api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
			
			resp = session.get(api_url, headers=headers, timeout=5)
			
			if resp.status_code == 200:
				results.append(resp.json().get("extract"))
			else:
				# Log status code to help debug if needed (optional)
				results.append(None)
		except:
			results.append(None)
	return pd.Series(results)

# --- UDF 2: Generic Scraper (Trafilatura) ---
@pandas_udf("string")
def fetch_external_content(urls: pd.Series) -> pd.Series:
	import requests, trafilatura
	results = []
	session = requests.Session()
	headers = {"User-Agent": "LandmarkResearchBot/1.0"}

	for url in urls:
		if not url or not isinstance(url, str):
			results.append(None)
			continue
		try:
			# Allow redirects for external sites
			r = session.get(url, headers=headers, timeout=(5, 10), allow_redirects=True)
			if r.status_code >= 400 or not r.text:
				results.append(None)
				continue
			
			# Extract main text
			text = trafilatura.extract(r.text)
			results.append(text[:5000] if text else None)
		except Exception:
			results.append(None)
	return pd.Series(results)

# --- Main Pipeline ---

# 1. Read CSV
print("📂 Reading CSV...")
df = spark.read.csv("enwikipedia_only_landmarks.csv", header=True)

# 2. Fetch (Safe Mode: 1000 partitions)
print("🚀 Starting Fetch (1000 partitions)...")
df_fetched = df.repartition(1000).withColumn(
	"fetched_text",
	when(col("website").contains("wikipedia.org"), fetch_wikipedia_summary(col("website")))
	.otherwise(fetch_external_content(col("website")))
)

# 3. Format Columns
final_df = df_fetched.select(
	col("qid"),
	col("qidLabel"),
	col("countryLabel"),
	col("lat"), 
	col("lon"),
	col("website").alias("website"),
	col("website").alias("final_url"),
	lit("en").alias("lang"),
	col("fetched_text").alias("text")
).withColumn(
	"status", 
	when(col("text").isNotNull() & (col("text") != ""), "ok").otherwise("error")
).withColumn(
	"doc_id", sha2(col("qid"), 256)
)

# 4. Save (Overwrite old bad data)
# Note: We save ALL rows (success + error) so you can verify counts later
output_path = os.path.join(os.getcwd(), "landmarks_ready_for_ollama")
print(f"💾 Saving to: {output_path}")

final_df.write.mode("overwrite").parquet(output_path)

print("✅ Job Complete.")