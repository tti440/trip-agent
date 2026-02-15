import os
import dotenv
from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
import hashlib
from neo4j import GraphDatabase
# 1. Setup
dotenv.load_dotenv("neo4j_acc.txt")
URI = os.getenv("NEO4J_LOCAL_URI")
USER = os.getenv("NEO4J_LOCAL_USER")
PWD = os.getenv("NEO4J_LOCAL_PASSWORD")
DATABASE = os.getenv("NEO4J_LOCAL_DATABASE")

graph = Neo4jGraph(url=URI, username=USER, password=PWD, database=DATABASE)
driver = GraphDatabase.driver(URI, auth=(USER, PWD))
emb = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# 2. Initialize Splitter
text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)

# 3. Initialize Graph Transformer (The Brain) 🧠
# We define a schema to keep the graph clean.
llm = ChatOllama(model="llama3", temperature=0, base_url="http://localhost:11434", format="json", num_ctx=2048)
llm_transformer = LLMGraphTransformer(
	llm=llm,
)

# 4. Your Manual Data
big_ben_data = {
	'qid': 'Q41225',
	'qidLabel': 'Big Ben',
	'countryLabel': 'United Kingdom',
	'lat': '51.50067',
	'lon': '-0.12457',
	'website': 'https://www.parliament.uk/bigben/',
	'final_url': 'https://www.parliament.uk/bigben/',
	'lang': 'en',
	'source_tag': 'manual_single_shot',
	'text': """
Big Ben is one of the world’s most famous landmarks in London, instantly recognisable and among the most photographed sites globally. The great clock and its bells have marked time over Westminster for more than 160 years, witnessing the reigns of six monarchs and the leadership of 41 prime ministers. Officially housed within the Elizabeth Tower, the structure stands as a powerful symbol of both the United Kingdom and democracy itself.

Built in the Victorian era to exceptionally high standards using the finest materials and craftsmanship, the tower has endured war damage, pollution, and harsh weather. Over time, however, its age created serious structural challenges. From 2017 to 2022, the Elizabeth Tower underwent the largest and most complex conservation project in its history. The restoration repaired every part of the building, from the gilded cross at the top to the base of its 334-step staircase, while also modernising facilities to ensure the tower remains functional for future generations.

Hundreds of specialist craftspeople contributed using traditional skills such as stone masonry, glass cutting, gilding, and horology. This investment preserved both the tower and Britain’s architectural heritage.

Standing 96 metres tall with 11 floors, the tower features 292 steps to the clock faces and 334 to the Belfry, where the Great Bell hangs. It is set to reopen to visitors, continuing its role as a treasured historic monument.
""",
	'status': 'error',
	'doc_id': '04606d91049d6b083428e12050159a27ee993441dc9bcf8141d6df440c0c3002'
}

def ingest_big_ben():
	print(f"🚀 Starting Ingestion for: {big_ben_data['qidLabel']}")
	
	# --- PHASE 1: CLEANUP & VECTOR INGESTION ---
	print("🧹 Cleaning old nodes...")
	graph.query(
		"MATCH (d:Document) WHERE d.qid = $qid OR d.qidLabel = $label DETACH DELETE d",
		{"qid": big_ben_data['qid'], "label": big_ben_data['qidLabel']}
	)

	print("embedding...")
	raw_doc = Document(page_content=big_ben_data['text'])
	chunks = text_splitter.split_documents([raw_doc])
	
	for i, chunk in enumerate(chunks):
		vector = emb.embed_query(chunk.page_content)
		unique_string = f"{big_ben_data['doc_id']}_{i}_{chunk.page_content[:20]}"
		chunk_hash = hashlib.md5(unique_string.encode()).hexdigest()

		# Create Document Node with ALL properties required by the schema
		graph.query(
			"""
			MERGE (d:Document {id: $chunk_hash})
			SET d.doc_id = $doc_id,
				d.text = $text,
				d.embedding = $vector,
				d.qid = $qid,
				d.qidLabel = $qidLabel,
				d.source_tag = $source_tag,  // Vital for the next phase
				d.website = $website,
				d.chunk_index = $index
			
			// Link to the main Entity (Big Ben)
			WITH d
			MERGE (e:Entity {qid: $qid})
			ON CREATE SET e.id = $qidLabel  // Ensure Entity exists
			MERGE (d)-[:MENTIONS]->(e)
			""",
			{
				"chunk_hash": chunk_hash,
				"doc_id": big_ben_data['doc_id'],
				"text": chunk.page_content,
				"vector": vector,
				"qid": big_ben_data['qid'],
				"qidLabel": big_ben_data['qidLabel'],
				"source_tag": big_ben_data['source_tag'],
				"website": big_ben_data['website'],
				"index": i
			}
		)
	print(f"✅ Created {len(chunks)} Document chunks.")

	# --- PHASE 2: METADATA ATTACHMENT ---
	print("🔗 Attaching Graph Metadata (Country, Location, Coords)...")
	graph_docs = llm_transformer.convert_to_graph_documents(chunks)
	graph.add_graph_documents(graph_docs)
	for doc in graph_docs[0]:
		print(doc)
		print("-----")
	print("🔗 Phase 3: Attaching Schema Metadata...")
	
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
	# Prepare the row object exactly as the Cypher expects it
	row = {
		"qid": big_ben_data['qid'],
		"source_tag": big_ben_data['source_tag'],
		"country": big_ben_data['countryLabel'],
		"coordinates": {
			"lat": float(big_ben_data['lat']), 
			"lon": float(big_ben_data['lon'])
		}
	}

	# Execute the attachment logic
	with driver.session(database=DATABASE) as session:
		session.run(ATTACH_METADATA_CYPHER, rows=[row])
		
	print("✅ Metadata attached successfully!")
	driver.close()

if __name__ == "__main__":
	ingest_big_ben()