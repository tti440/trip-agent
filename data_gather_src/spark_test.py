import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import functions as F
import requests
import time
import os
import dotenv

# 1. SETUP & CONFIG
dotenv.load_dotenv("neo4j_acc.txt")
URI = os.getenv("NEO4J_LOCAL_URI") or "bolt://localhost:7687"
USER = os.getenv("NEO4J_LOCAL_USER") or "neo4j"
PWD = os.getenv("NEO4J_LOCAL_PASSWORD") or "password"

spark = SparkSession.builder \
    .appName("WikiFillTextAndURL") \
    .config("spark.jars.packages", "org.neo4j:neo4j-connector-apache-spark_2.12:5.3.2_for_spark_3") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# 2. WORKER FUNCTION (Runs on Executors)
def fetch_wiki_data_batch(iterator):
    # Reuse session for speed
    session = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    headers = {'User-Agent': 'GraphHealer/SparkBot (your_email@example.com)'}
    
    output_rows = []
    
    for row in iterator:
        qid = row.id
        label = row.label
        
        # Skip if no label
        if not label: continue

        try:
            # Combined Query: Get Text AND URL in one shot
            params = {
                "action": "query",
                "format": "json",
                "titles": label,
                "prop": "extracts|info", # Get Extract AND Page Info
                "inprop": "url",         # Specifically ask for the URL
                "exintro": 1,            # Intro only
                "explaintext": 1,        # Clean text
                "redirects": 1           # Follow "Mt Fuji" -> "Mount Fuji"
            }
            
            resp = session.get(url, params=params, headers=headers, timeout=5)
            data = resp.json()
            
            pages = data.get("query", {}).get("pages", {})
            
            # Find the valid page (ignoring "-1" which means missing)
            found_text = None
            found_url = None
            
            for pid, pdata in pages.items():
                if pid == "-1": continue
                
                # 1. Get Text
                txt = pdata.get("extract", "")
                if "may refer to:" in txt or len(txt) < 50:
                    continue
                
                # 2. Get URL
                page_url = pdata.get("fullurl", "")
                
                # Clean up text
                clean_text = txt.strip().replace("\n", " ").replace("\\", "")
                
                found_text = clean_text
                found_url = page_url
                break
            
            # Only yield if we found good data
            if found_text and found_url:
                output_rows.append((qid, found_text, found_url))
            
            # Rate limit per core
            time.sleep(0.1)
            
        except Exception:
            pass
            
    return iter(output_rows)

# 3. MAIN EXECUTION
if __name__ == "__main__":
    
    # A. Read Input
    # Assuming CSV has columns: id, label, URL, length, d.text
    df = spark.read.csv("short_text_qid.csv", header=True)
    
  
    
    # Repartition for safe concurrency (e.g. 10 workers)
    df_processing = df.repartition(10)
    
    # B. Run Fetcher
    rdd_result = df_processing.rdd.mapPartitions(fetch_wiki_data_batch)
    
    # Create DataFrame from results
    schema = StructType([
        StructField("qid", StringType(), True),
        StructField("new_text", StringType(), True),
        StructField("new_url", StringType(), True)
    ])
    
    df_healed = spark.createDataFrame(rdd_result, schema)
    
    # Show what we found
    df_healed.show(5, truncate=50)
    
    # C. Write to Neo4j (Update nodes)
    print("💾 Writing updates to Neo4j...")
    
    df_healed.write \
        .format("org.neo4j.spark.DataSource") \
        .mode("Append") \
        .option("url", URI) \
        .option("authentication.type", "basic") \
        .option("authentication.basic.username", USER) \
        .option("authentication.basic.password", PWD) \
        .option("query", """
            MATCH (d:Document {qid: event.qid})
            SET d.text = event.new_text,
                d.URL  = event.new_url
        """) \
        .save()
        
    print("✅ Success! Text and URLs have been backfilled.")
    spark.stop()