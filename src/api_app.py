from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from tools import rt_tools
import json
import requests
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import shutil
import uvicorn
import re

BACKEND_HOST = os.getenv("BACKEND_SERVICE_HOST", "backend") 
app_api = FastAPI(title="Landmark Description and Gather Data Service")

def clean_text(output_descriptions: str, max_retries: int = 5) -> str:
    match = re.search(r'\{.*\}', output_descriptions, re.DOTALL)

    if match:
        potential_json = match.group(0)
        try:
            json.loads(potential_json)
            return potential_json
        except:
            pass 

    llm = ChatOllama(model="llama3", temperature=0, num_ctx=2048) 
    prompt = f"Fix this JSON so it parses correctly. Return ONLY valid JSON:\n{output_descriptions}"
    print("🛠️ JSON FIXER: Cleaning output with LLM...")
    for attempt in range(max_retries):
        print(f"Attempt {attempt+1} to clean JSON...")
        response = llm.invoke(prompt)
        try:
            json.loads(response.content)
            print("✅ JSON cleaned successfully.")
            return response.content
        except:
            print("⚠️ Cleaned JSON is still invalid. Retrying...")
    print("❌ Failed to clean JSON after multiple attempts. Returning original output.")
    return "{}"

# def clean_text(output_descriptions: str) -> str:
# 	i=0
# 	new_text = "{\n"
# 	while (i<len(output_descriptions)):
# 		j = output_descriptions.find("}",i)
# 		line = output_descriptions[output_descriptions.find("\"",i)-1:output_descriptions.find("}",j)+1]
# 		line = line.strip()
# 		tmp_text = line[0]
# 		for index in range(1,len(line)-1):
# 			if line[index-1] == "\"" and line[index]==" " and line[index+1]=="\n":
# 				continue
# 			elif line[index-1] == " " and line[index]==" " and line[index+1]==" ":
# 				continue
# 			elif line[index-1] == "\"" and line[index]==" " and line[index+1]==" ":
# 				continue
# 			elif line[index-1] == " " and line[index]==" " and line[index+1]=="\n":
# 				continue
# 			else:
# 				tmp_text += line[index]
# 		new_text += tmp_text
# 		new_text += "},"
# 		new_text += "\n"
# 		i=j+1
# 	new_text = new_text[:-2]
# 	new_text += "}"
# 	return new_text

class CandidateState(TypedDict):
    image_path: str
    caption: str
    text_input: str
    landmark_name: str
    landmark_description: str
    history_data: str
    logistics_data: str
    cultural_data: str
    accommodation_data: str
    food_data: str
    final_plan: str
    itineraries: List[str]

def skip_update(current_state, new_state):
    return current_state if current_state else new_state

class TripState(TypedDict, total=False):
    image_path: Annotated[str, skip_update]
    caption: Annotated[str, skip_update]
    itineraries: Annotated[List[str], operator.add]
    candidates_raw: dict
    text_input: Annotated[str, skip_update]
    
def core_handler(state):
    print("🧠 CORE AGENT: Identifying Landmark...")
    image_path = state["image_path"]
    text_input = state.get("text_input", "")
    backend_url = f"http://{BACKEND_HOST}:8000/identify?image_path={image_path}&text_input={text_input}"
    try:
        response_obj = requests.get(backend_url)
        response_obj.raise_for_status()
        response = response_obj.json()
        print(f"DEBUG: Backend response: {response}")
        graph_desc = response["candidates_raw"]
        caption = response["caption"]
        #graph_desc=graph_desc[graph_desc.find("{"):graph_desc.rfind("}")].strip()
        graph_desc= clean_text(graph_desc.strip())
        data = json.loads(graph_desc)
        print("Graph description JSON parsed successfully.")
        with open("graph_description.json", "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error fetching or parsing backend response: {str(e)}")
        print("No valid JSON found in graph description.")
        data = {}
    return {
        "caption": caption, 
        "candidates_raw": data,
        "image_path": image_path
    }

def history_handler(state):
    name = state["landmark_name"]
    desc = state["landmark_description"]
    target = f"{name} {desc}"
    print(f"📜 HISTORY: Web search for {target}...")
    return {"history_data": rt_tools.get_history(target)}

def logistics_handler(state):
    name = state["landmark_name"]
    desc = state["landmark_description"]
    target = f"{name} {desc}"
    print(f"🚌 LOGISTICS: Transport search for {target}...")
    return {"logistics_data": rt_tools.get_logistics(target)}

def culture_handler(state):
    name = state["landmark_name"]
    desc = state["landmark_description"]
    target = f"{name} {desc}"
    print(f"🎎 CULTURE: Etiquette search for {target}...")
    return {"cultural_data": rt_tools.get_culture(target)}

def accommodation_handler(state):
    name = state["landmark_name"]
    desc = state["landmark_description"]
    target = f"{name} {desc}"
    print(f"🏨 ACCOMMODATION: Hotel search for {target}...")
    return {"accommodation_data": rt_tools.get_accommodation(target)}

def food_handler(state):
    name = state["landmark_name"]
    desc = state["landmark_description"]
    target = f"{name} {desc}"
    print(f"🍜 FOOD: Restaurant search for {target}...")
    return {"food_data": rt_tools.get_food(target)}

def writer_handler(state):
    print("✍️ WRITER: Synthesizing Final Itinerary...")
    
    llm = ChatOllama(model="llama3.2:3b", temperature=0.7, num_ctx=8192)
    
    prompt = ChatPromptTemplate.from_template("""
    You are an expert Travel Planner.
    
    LANDMARK: {landmark}
    
    1. USER VISUAL CONTEXT: {caption}
    2. USER TEXT INPUT: {text_input}
    3. NEO4J FACTS: {graph_desc}
    4. WEB HISTORY: {history}
    5. TRANSPORT: {logistics}
    6. CULTURE/TIPS: {culture}
    7. HOTELS: {hotels}
    8. FOOD: {food}
    
    TASK:
    Write a travel guide for this location.
    Organize it into clear sections with emojis.
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "caption": state["caption"],
        "text_input": state["text_input"],
        "landmark": state["landmark_name"],
        "graph_desc": state["landmark_description"],
        "history": state["history_data"],
        "logistics": state["logistics_data"],
        "culture": state["cultural_data"],
        "hotels": state["accommodation_data"],
        "food": state["food_data"]
    })
    print("Final Itinerary Generated.")
    return {"itineraries": [response]}

def map_candidates(state):
    raw_data = state.get("candidates_raw", {})
    caption = state.get("caption", "")
    image_path = state.get("image_path", "")
    text_input = state.get("text_input", "")
    tasks = []
    
    for key, val in raw_data.items():
        # Build the exact input for each parallel worker
        candidate_input = {
            "image_path": image_path,
            "caption": caption,
            "text_input": text_input,
            "landmark_name": val.get('name', 'Unknown'),
            "landmark_description": val.get('description', ''),
            "itineraries": [] # Initialize an empty list for this worker
        }
        # Send starts the parallel subgraph instances
        tasks.append(Send("landmark_processor", candidate_input))
        
    return tasks

candidate_workflow = StateGraph(CandidateState)

candidate_workflow.add_node("history", history_handler)
candidate_workflow.add_node("logistics", logistics_handler)
candidate_workflow.add_node("culture", culture_handler)
candidate_workflow.add_node("accommodation", accommodation_handler)
candidate_workflow.add_node("food", food_handler)
candidate_workflow.add_node("writer", writer_handler)

# Edges: Start -> Parallel Tools -> Writer -> End
candidate_workflow.add_edge(START, "history")
candidate_workflow.add_edge(START, "logistics")
candidate_workflow.add_edge(START, "culture")
candidate_workflow.add_edge(START, "accommodation")
candidate_workflow.add_edge(START, "food")

candidate_workflow.add_edge("history", "writer")
candidate_workflow.add_edge("logistics", "writer")
candidate_workflow.add_edge("culture", "writer")
candidate_workflow.add_edge("accommodation", "writer")
candidate_workflow.add_edge("food", "writer")

candidate_workflow.add_edge("writer", END)
candidate_app = candidate_workflow.compile()

workflow = StateGraph(TripState)

# Nodes
workflow.add_node("core", core_handler)
workflow.add_node("landmark_processor", candidate_app)

# Edges
workflow.add_edge(START, "core")

workflow.add_conditional_edges(
    "core", 
    map_candidates, 
    ["landmark_processor"]
)
workflow.add_edge("landmark_processor", END)

app = workflow.compile()

@app_api.post("/plan")
async def generate_plan(file: UploadFile = File(...), text_input: str = Form("")):
    import datetime
    start = datetime.datetime.now()
    temp_path = f"user_img/{file.filename}"
    os.makedirs("user_img", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        inputs = {"image_path": temp_path, "text_input": text_input, "itineraries": []}
        
        print("🚀 Starting Multi-Candidate Analysis...")
        result = app.invoke(inputs)
        
        print("\n" + "="*50)
        print(f"✅ Generated {len(result['itineraries'])} Plans:\n")
        
        for plan in result['itineraries']:
            print(plan)
            print("\n" + "-"*30 + "\n")
        end = datetime.datetime.now()
        print(f"Total time taken: {end - start}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during plan generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    return {
    "status": "success",
    "itineraries": result['itineraries'],
    "count": len(result['itineraries'])
            }

if __name__ == "__main__":
    uvicorn.run(app_api, host="0.0.0.0", port=8001)