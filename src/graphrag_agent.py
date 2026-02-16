from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
import os
import dotenv
import re
import math
import subprocess

cmd = "ip route show | grep default | awk '{print $3}'"
WINDOWS_HOST_IP = subprocess.getoutput(cmd)

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
DATABASE = os.getenv("NEO4J_DATABASE") 

graph = Neo4jGraph(url=URI, username=USER, password=PWD, database=DATABASE)

graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id, e.qid, e.qidLabel]")

emb = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=f"http://ollama-service:11434",
    num_ctx=2048,
)

vector_index = Neo4jVector.from_existing_index(
    url=URI,
    username=USER,
    password=PWD,
    database=DATABASE,
    embedding=emb,
    index_name="vector",          # Must match your CREATE VECTOR INDEX name
    keyword_index_name="keyword",
    text_node_property="text",    # The property containing the text
    search_type="hybrid",
    
)

entity_model = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    base_url=f"http://ollama-service:11434",
    num_ctx=2048)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    base_url=f"http://ollama-service:11434",
    num_ctx=8192
)

class Entities(BaseModel):
    names: List[str] = Field(
        ...,
        description=(
            "Extract specific distinct geographical locations. "
            "Focus ONLY on Proper Nouns, Cities, Countries, and Regions (e.g., 'London', 'UK', 'Tuscany'). "
            "DO NOT extract generic terms like 'beach', 'city', 'downtown', 'tower', or 'street'."
        ),
    )

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise Named Entity Recognition (NER) system. Extract ONLY specific locations."),
    ("human", "input: {question}"),
])

entity_chain = prompt | entity_model.with_structured_output(Entities)

def remove_lucene_chars(text: str) -> str:
    return re.sub(r'[+\-!(){}\[\]^"~*?:\\/]', ' ', text)

def generate_full_text_query(input: str) -> str:
    words = [el for el in remove_lucene_chars(input).split() if el]
    if not words: return ""
    # Use fuzzy matching (~2) for slight spelling variations
    return " AND ".join([f"{word}~2" for word in words])

def doc_qid(doc):
    md = getattr(doc, "metadata", {}) or {}
    return md.get("qid"), md.get("qidLabel")

def top_landmark_candidates(docs_with_scores, target_locations, topn=10):
    """
    Ranks candidates by Similarity Score, applying a 1.1x BOOST 
    if the candidate is geographically located in the target area.
    """
    best_scores = {}
    labels = {}
    
    # 1. Get unique QIDs to batch query
    candidate_qids = list(set([d.metadata.get("qid") for d, _ in docs_with_scores if d.metadata.get("qid")]))
    
    boost_qids = set()
    node_importance = {}
    
    if target_locations:
        validation_query = """
        UNWIND $qids AS candidate_qid
        CALL (candidate_qid) {
        MATCH (doc:Document {qid: candidate_qid})
        RETURN doc AS target, doc.qid AS output_qid

        UNION

        MATCH (doc:Document {qid: candidate_qid})
        MATCH (entity:__Entity__)
        WHERE entity.id = doc.qidLabel
        RETURN entity AS target, doc.qid AS output_qid
        }

        WITH output_qid, target,
            COUNT { (target)<--() } AS in_degree,
            COUNT { (target)-->() } AS out_degree,
            EXISTS {
            MATCH p = (target)-[*1..2]->(loc)
            WHERE (loc.id IN $targets OR loc.qidLabel IN $targets)
                AND any(r IN relationships(p) WHERE coalesce(r.rel_group,'') = 'LOCATED_IN')
            } AS in_target

        WITH output_qid,
            max(in_degree) AS in_degree,
            max(out_degree) AS out_degree,
            max(in_target) AS is_in_target

        RETURN output_qid AS qid, in_degree, out_degree, is_in_target
        """
        
        try:
            rows = graph.query(validation_query, {
                "qids": candidate_qids, 
                "targets": target_locations if target_locations else []
            })
            
            for row in rows:
                weighted_degree = (row["in_degree"] * 2) + row["out_degree"] + 1
                node_importance[row["qid"]] = math.log(weighted_degree)
                
                # Geographic Match
                if row["is_in_target"]:
                    boost_qids.add(row["qid"])
        except Exception as e:
            print(f"⚠️ Boosting Query Failed: {e}")

    # 2. SCORING LOOP (Pure Python - Fast)
    for doc, score in docs_with_scores:
        qid, qlbl = doc_qid(doc)
        if not qid: continue
        final_score = score
        # APPLY BOOST
        if qid in boost_qids:
            print(f"🚀 Boosting {qlbl} ({qid}) for geographic relevance {target_locations}.")
            final_score *= 1.1
        importance = node_importance.get(qid, 0)
        final_score += (importance * 0.05)

        # STRATEGY: MAX SCORE
        if qid not in best_scores or final_score > best_scores[qid]:
            best_scores[qid] = final_score
            labels[qid] = qlbl
            
    # 3. RANKING
    ranked = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    
    return [(qid, labels[qid], score) for qid, score in ranked]

# from collections import Counter
# def top_landmark_candidates(docs, topn=10):
#     # Instead of just counting frequency (1, 2, 3...), we sum a "relevance score".
#     # Rank 1 gets 1.0 points. Rank 100 gets 0.01 points.
#     scores = {}
#     labels = {}
    
#     for i, d in enumerate(docs):
#         qid, qlbl = doc_qid(d)
#         if not qid: continue
        
#         # Scoring Formula: Higher rank (lower i) = More points
#         # 1st place = 100 pts, 2nd = 99 pts...
#         score = 100 - i 
        
#         if qid not in scores:
#             scores[qid] = 0
#             labels[qid] = qlbl
        
#         scores[qid] += score

#     # Sort by total score (High to Low)
#     ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topn]
    
#     return [(qid, labels[qid], score) for qid, score in ranked]

# def top_landmark_candidates(docs, topn=15):
#     c = Counter()
#     label = {}
#     for d in docs:
#         qid, qlbl = doc_qid(d)
#         if qid:
#             c[qid] += 1
#             if qlbl:
#                 label[qid] = qlbl
#     return [(qid, label.get(qid), cnt) for qid, cnt in c.most_common(topn)]

def fetch_landmark_graph_context(candidates, per=40):
    qids = [qid for qid, _, _ in candidates if qid]
    if not qids:
        return "No landmark candidates (qid) found from vector hits."

    rows = graph.query(
        """
        UNWIND $qids AS qid
        MATCH (doc:Document {qid: qid})
        OPTIONAL MATCH (ent:__Entity__ {id: doc.qidLabel})

        WITH qid, [x IN [doc, ent] WHERE x IS NOT NULL] AS starts
        UNWIND starts AS l

        CALL (l){
        WITH l
        MATCH (l)-[r:REL]-(n)
        WHERE coalesce(r.rel_group,'') <> 'OTHER'
        WITH l, r, n
        LIMIT $per
        RETURN collect(
            coalesce(l.qidLabel,l.id,'?') + ' --[' +
            coalesce(r.rel_group,r.rel_type_en,r.rel_type,'REL') + ']-- ' +
            coalesce(n.qidLabel,n.id,n.qid,left(n.text,80),'?')
        ) AS facts
        }

        RETURN qid AS qid,
            coalesce(l.qidLabel,l.id) AS label,
            facts AS facts
        """,
        {"qids": qids, "per": per},
    )

    out = []
    for r in rows:
        out.append(f"== Candidate {r['label']} ({r['qid']}) ==\n" + "\n".join(r["facts"]))
    return "\n\n".join(out) if out else "No graph context found for candidates."

# def retriever(question: str) -> str:
#     print(f"🚀 Retrieving for: {question}")
    
#     # 1. Graph Search
#     structured_data = structured_retriever(question)
    
#     # 2. Vector Search (Simpler invocation)
#     # The 'k=50' ensures we get 50 closest documents
#     unstructured_docs = vector_index.similarity_search(question, k=50)
#     unstructured_data = "\n".join([f"DOC: {d.page_content}" for d in unstructured_docs])
    
#     print(f"📊 Stats: {len(unstructured_docs)} vector docs found.")

#     return f"""
#     Structured Knowledge:
#     {structured_data}

#     Unstructured Documents:
#     {unstructured_data}
#     """

def retriever(question: str, caption: str = None, keywords: set[str] = None) -> str:
    # --- 1. SEARCH STRATEGY: PURE VISUAL SIGNAL ---
    
    if caption and keywords:
        kw_str = " ".join(sorted(set(keywords)))
        retrieval_query = f"{caption} {kw_str}"
        retrieval_query = retrieval_query.replace(",", " ")
    else:
        retrieval_query = question
    
    print("🧠 Running NER...")
    try:
        entities = entity_chain.invoke({"question": retrieval_query})
        exclude_set = {"city", "region", "country"}
        target_locations = [e for e in entities.names if e.lower() not in exclude_set] # e.g. ['London']
        print(f"📍 Extracted Locations: {target_locations}")
        for location in target_locations:
            retrieval_query += f" {location}"
    except Exception as e:
        print(f"⚠️ NER Error: {e}")
        target_locations = []
    # print(target_locations)
    print(f"🚀 Vector retrieval query: {retrieval_query}")
    vec_docs = vector_index.similarity_search_with_score(retrieval_query, k=500)
    with open("debug_vec_docs.pkl", "wb") as f:
        import pickle
        pickle.dump(vec_docs, f)
    
    allowed_location_ids = set()
    if target_locations:
        # print(f"🌍 Expanding Region for targets: {target_locations}")
        
        expansion_query = """
        MATCH (n:__Entity__)
        WHERE n.id IN $targets OR n.qidLabel IN $targets
        WITH n


        MATCH (n)-[r1]-(d)
        WHERE r1.rel_group = "LOCATED_IN" 
        AND (d:City OR d:Country OR d:Region)
        WITH n, d

        OPTIONAL MATCH (d:Country)-[r2]-(c:City)
        WHERE r2.rel_group = "LOCATED_IN" 
        AND c <> n 
        
        RETURN 
            n.id AS n_id, n.qidLabel AS n_lbl,
            d.id AS d_id, d.qidLabel AS d_lbl,
            c.id AS c_id, c.qidLabel AS c_lbl
        """
        
        try:
            region_rows = graph.query(expansion_query, {"targets": target_locations})
            
            # Flatten results into the allowed set
            for row in region_rows:
                # Add Target
                if row["n_id"]: allowed_location_ids.add(row["n_id"])
                if row["n_lbl"]: allowed_location_ids.add(row["n_lbl"])
                
                # Add Container (Country)
                if row["d_id"]: allowed_location_ids.add(row["d_id"])
                if row["d_lbl"]: allowed_location_ids.add(row["d_lbl"])
                
                # Add Sibling (City)
                if row["c_id"]: allowed_location_ids.add(row["c_id"])
                if row["c_lbl"]: allowed_location_ids.add(row["c_lbl"])
                
            print(f"✅ Allowed Region Size: {len(allowed_location_ids)} locations")
            print(f"   (Includes: {list(allowed_location_ids)}...)")
            
        except Exception as e:
            print(f"⚠️ Expansion Error: {e}")

    final_docs = []
    if allowed_location_ids:
        print(f"🛡️ Applying Graph Filter for: {target_locations}")
        candidate_qids = [d.metadata.get("qid") for d, _ in vec_docs if d.metadata.get("qid")]
        # Get all QIDs from the 500 candidates
        filter_query = """
        UNWIND $qids AS qid
        MATCH (doc:Document {qid: qid})
        OPTIONAL MATCH (ent:__Entity__ {id: doc.qidLabel})
        WITH qid, [x IN [doc, ent] WHERE x IS NOT NULL] AS starts
        UNWIND starts AS l
        WITH DISTINCT qid, l

        WHERE EXISTS {
        MATCH p = (l)-[*1..2]->(loc)
        WHERE (loc.id IN $allowed OR loc.qidLabel IN $allowed)
            AND any(rel IN relationships(p)
                    WHERE coalesce(rel.rel_group,'') = 'LOCATED_IN')
        }

        RETURN DISTINCT qid AS valid_qid
        """
        
        try:
            valid_rows = graph.query(filter_query, {
                "qids": candidate_qids,
                "allowed": list(allowed_location_ids)
            })
            valid_qids = set(row["valid_qid"] for row in valid_rows)
            # print(f"🛡️ Filter kept {len(valid_qids)} / {len(vec_docs)} candidates.")
            
            final_docs = [d for d in vec_docs if d[0].metadata.get("qid") in valid_qids]
        except Exception as e:
            # print(f"❌ Filter Failed: {e}")
            final_docs = vec_docs[:50]
            
    else:
        final_docs = vec_docs[:50]

    # Fallback
    if not final_docs:
        # print("⚠️ Filter too strict. Showing top 10 raw results.")
        final_docs = vec_docs[:50]
            
    candidates = top_landmark_candidates(final_docs, target_locations,topn=25)
    candidate_graph = fetch_landmark_graph_context(candidates, per=20)
    qid_to_doc = {}
    for d, score in final_docs:
        qid = d.metadata.get("qid")
        if qid:
            # If multiple docs have the same QID, keep the longest/best one
            if qid not in qid_to_doc or len(d.page_content) > len(qid_to_doc[qid]):
                qid_to_doc[qid] = d.page_content

    evidence_list = []
    for qid, label, score in candidates[:25]:
        text = qid_to_doc.get(qid, "No text available.")
        # Clean up newlines for cleaner prompt
        clean_text = text[:700].replace("\n", " ")
        evidence_list.append(f"== {label} (Score={score:.4f}) ==\n{clean_text}...")

    doc_snips = "\n\n".join(evidence_list)
    cand_list = "\n".join([f"- {lbl} ({qid}) Score={score:.4f}" for qid, lbl, score in candidates])
    with open("output-candidate.txt", "w") as f:
        f.write(f"CANDIDATES:\n{cand_list}\n\nGRAPH CONTEXT:\n{candidate_graph}\n\nEVIDENCE:\n{doc_snips}")
    return f"CANDIDATES:\n{cand_list}\n\nGRAPH CONTEXT:\n{candidate_graph}\n\nEVIDENCE:\n{doc_snips}"

# --- Main Chain ---

template = """You are an expert travel investigator.
Based on the provided "CANDIDATE LANDMARKS" (which are ranked by relevance score) and the "GRAPH CONTEXT", 
identify the TOP 3 most likely candidates matching the user context.

Context:
{context}

Question:
{question}
"""

def context_from_inputs(inputs: dict) -> str:
    # inputs contains {"caption":..., "keywords":..., "question":...}
    # We pass these to your existing retriever function
    return retriever(
        question=inputs["question"],
        caption=inputs.get("caption"),
        keywords=inputs.get("keywords")
    )

chain = (
    RunnableParallel({
        "context": context_from_inputs,
        "question": lambda inputs: inputs["question"],
    })
    | ChatPromptTemplate.from_template(template)
    | llm
    | StrOutputParser()
)

def build_multimodal_question(caption: str, keywords: set[str], user_input: str) -> str:
    kw_list = list(keywords)
    # This string is for the LLM to read and understand the task
    return f"""
    PRIMARY VISUAL DESCRIPTION:
    {caption}
    
    AUXILIARY CONTEXT (Keywords from similar images):
    {', '.join(kw_list)}
    
    USER INPUT:
    {user_input}
    
    USER REQUEST:
    Analyze the Visual Description and Context and USER INPUT. 
    Rank the Top 3 best matching landmarks from the candidates provided.
    For each, explain why it fits the description ("{caption}") in JSON format.
    Respond EXACTLY in the following JSON format:
    {{
        "1":
            {{
            "name": "Landmark Name",
            "description": "Description of the landmark",
            "reasoning": "Explanation of why this landmark matches the visual description."}},
        "2": {{
            "name": "Landmark Name",
            "description": "Description of the landmark",
            "reasoning": "Explanation of why this landmark matches the visual description."}},
        "3": {{
            "name": "Landmark Name",
            "description": "Description of the landmark",
            "reasoning": "Explanation of why this landmark matches the visual description."}}
        }}
    """
    #     "4": {{
    #         "name": "Landmark Name",
    #         "description": "Description of the landmark",
    #         "reasoning": "Explanation of why this landmark matches the visual description."}},
    #     "5": {{
    #         "name": "Landmark Name",
    #         "description": "Description of the landmark",
    #         "reasoning": "Explanation of why this landmark matches the visual description."}}
    # }}
    # """

def graphrag_run(caption: str, keywords: list[str], user_input: str) -> str:
    question = build_multimodal_question(caption, keywords, user_input)
    answer = chain.invoke({"question": question, "caption": caption, "keywords": keywords})
    return answer

# def main():
#     caption = "A clock tower in a city at night"
#     keywords = ["clock tower", "bell", "London", "historic landmark"]
#     result = graphrag_run(caption, keywords)
#     print(result)