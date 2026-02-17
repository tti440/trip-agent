from graphrag_agent import graphrag_run
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import faiss
from graphrag_agent import graphrag_run
import os
from langchain_ollama import ChatOllama
from service_url import BACKEND_URL, ORCHESTRATOR_URL, OLLAMA_URL


filedir = os.path.dirname(os.path.abspath(__file__))

class VisionKnowledgeBackend:
	_instance = None
	def __new__(cls):
		if cls._instance is None:
			cls._instance = super(VisionKnowledgeBackend, cls).__new__(cls)
			cls._instance._initialize_models()
		return cls._instance

	def _initialize_models(self):
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
		self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device)
		self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
		self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
		self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
		try:
			self.image_index = faiss.read_index("clip_image.index")
			self.text_index = faiss.read_index("clip_text.index")
			self.keywords_df = pd.read_csv("keywords.csv")
			self.ids_list = pd.read_csv("photo_ids.csv", header=None)
		except Exception as e:
			self.image_index = faiss.read_index(os.path.join(filedir, "../data", "clip_image.index"))
			self.text_index = faiss.read_index(os.path.join(filedir, "../data", "clip_text.index"))
			self.keywords_df = pd.read_csv(os.path.join(filedir, "../data", "keywords.csv"))
			self.ids_list = pd.read_csv(os.path.join(filedir, "../data", "photo_ids.csv"), header=None)
		# print("✅ Models and indexes loaded.")

	def image_to_text(self, image_path):
		image = Image.open(image_path).convert("RGB")
		blip_inputs = self.blip_processor(images=image, return_tensors="pt").to(self.device)
		blip_outputs = self.blip_model.generate(**blip_inputs)
		text = self.blip_processor.batch_decode(blip_outputs, skip_special_tokens=True)[0]
		
		img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
		txt_inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
		img_feat = self.model.get_image_features(**img_inputs)
		txt_feat = self.model.get_text_features(**txt_inputs)
		img_feat = img_feat["pooler_output"] / img_feat["pooler_output"].norm(dim=-1, keepdim=True)
		txt_feat = txt_feat["pooler_output"] / txt_feat["pooler_output"].norm(dim=-1, keepdim=True)
		return text, img_feat, txt_feat

	def get_relevant_tags(self, image_description, keyword_list, top_k=10, threshold=0.35):
		query_embedding = self.embedding_model.encode(image_description, convert_to_tensor=True)
		tag_embeddings = self.embedding_model.encode(keyword_list, convert_to_tensor=True)
		cosine_scores = util.cos_sim(query_embedding, tag_embeddings)[0]
		top_results = torch.topk(cosine_scores, k=top_k)

		selected_tags = []
		
		for score, idx in zip(top_results.values, top_results.indices):
			tag = keyword_list[idx]
			score_val = score.item()
			
			if score_val > threshold:
				selected_tags.append(tag)

		return selected_tags

	def get_keywords_from_image(self, image_path):
		caption, img_feat, txt_feat = self.image_to_text(image_path)
		D_img, I_img = self.image_index.search(img_feat.detach().cpu().numpy(), k=self.image_index.ntotal)
		D_txt, I_txt = self.text_index.search(txt_feat.detach().cpu().numpy(), k=self.text_index.ntotal)

		img_scores = {int(idx): float(dist) for idx, dist in zip(I_img[0], D_img[0])}
		txt_scores = {int(idx): float(dist) for idx, dist in zip(I_txt[0], D_txt[0])}

		combined_scores = {}
		all_ids = set(img_scores.keys()).union(txt_scores.keys())

		for pid in all_ids:
			d1 = img_scores.get(pid)
			d2 = txt_scores.get(pid)
			
			if d1 is None:
				combined_scores[pid] = d2
			elif d2 is None:
				combined_scores[pid] = d1
			else:
				combined_scores[pid] = (d1*0.6+ d2*0.4)

		top_k = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:3]  # or [:3]
		return top_k, caption

	def process_keywords(self, image_path):
		top_k, caption = self.get_keywords_from_image(image_path)
		id_1 = self.ids_list.iloc[top_k[0][0]]
		id_2 = self.ids_list.iloc[top_k[1][0]]
		id_3 = self.ids_list.iloc[top_k[2][0]]
		keywords_1 = self.keywords_df[self.keywords_df["photo_id"] == id_1[0]]
		keywords_2 = self.keywords_df[self.keywords_df["photo_id"] == id_2[0]]
		keywords_3 = self.keywords_df[self.keywords_df["photo_id"] == id_3[0]]
		keywords = set()
		keywords.update(keywords_1["keyword"].values.tolist())
		keywords.update(keywords_2["keyword"].values.tolist())
		keywords.update(keywords_3["keyword"].values.tolist())
	
		keywords = self.get_relevant_tags(caption, list(keywords), top_k=15, threshold=0.35)
		return caption, keywords

	def _get_text_keywords(self, description: str, text_input: list[str]) -> list:
		if not text_input or len(text_input) == 0:
			return []
		llm = ChatOllama(model="llama3", temperature=0, num_ctx=2048, base_url=OLLAMA_URL)
		text_input = [kw.replace(".", "") for kw in text_input]
		prompt = f"""
				Identify the most meaningful keywords ONLY from the User Input that match the Visual Caption.
				Divide your output into two priorities:

				1. ANCHORS: Specific landmark names and locations (Proper Nouns).
				2. CONTEXT: Adjectives, moods, and styles found in the User Input (e.g., vibrant, historical, happy, urban, night-life).
				Output ONLY the keywords from BOTH categories as a SINGLE comma-separated list. 
    			DO NOT add explanations or extra texts. 
       
       			User Input: "{text_input}" 
          		Visual Caption: "{description}"
            """
		try:
			print(f"🧠 LLM: Extracting keywords from text: {text_input}")
			response = llm.invoke(prompt)
			print(f"🧠 LLM: Raw keyword extraction response: {response.content}")
			extracted = [kw.strip() for kw in response.content.replace("\n", ",").replace(":", ",").replace(";", ",").replace("*", ",").split(",")]
			print(f"🧠 LLM: Extracted keywords before filtering: {extracted}")
			tmp = [b.strip() for item in extracted for b in item.split("\n") if b.lower().strip() in [text.lower() for text in text_input]]
			print(f"🧠 LLM: Extracted keywords after filtering: {tmp}")
			return tmp
		except Exception as e:
			print(f"⚠️ Warning: LLM keyword extraction failed: {e}")
			return []
		
	def run_generation_pipeline(self, image_path, text_input=""):
		caption, keywords = self.process_keywords(image_path)
		text_input_list = text_input.strip().split(" ")
		keywords.extend(text_input_list)
		keywords = self._get_text_keywords(caption, keywords)
		# keywords = [k for kw in keywords for k in kw.split(" ")]
		keywords = list(set(keywords))
		import gc; gc.collect(); torch.cuda.empty_cache()
		# remove word people or person from keywords and captions
		keywords = [kw for kw in keywords if kw.lower() not in ["people", "person"]]
		print(f"🔍 Extracted Keywords: {keywords}")
		caption = caption.replace("people", "").replace("person", "").strip()
		result = graphrag_run(caption, keywords, text_input)
		return result, caption
	
backend = VisionKnowledgeBackend()
 
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--img_path", type=str, required=True, help="Path to the input image")
	parser.add_argument("--text_input", type=str, default="", help="User text input for additional context")
	args = parser.parse_args()
	import time
	start_time = time.time()
	description, caption = backend.run_generation_pipeline(args.img_path, args.text_input)
	# with open("output_description.txt", "w") as f:
	# 	f.write(description)
	print("Caption:", caption)
	print("Description:", description)
	print(f"Pipeline execution time: {time.time() - start_time:.2f} seconds")