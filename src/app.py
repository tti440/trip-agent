import streamlit as st
import requests
import os
import base64

st.set_page_config(page_title="AI Travel Planner", layout="wide")
st.title("🌍 AI Multimodal Travel Agent")


col1, col2 = st.columns(2)

with col1:
	uploaded_file = st.file_uploader("📸 Upload a landmark photo...", type=['jpg', 'jpeg', 'png'])
	if uploaded_file:
		st.image(uploaded_file, caption="Target Landmark", use_container_width=True)
with col2:
	user_text = st.text_area("📝 Additional context or preferences?", 
							 placeholder="e.g. 'I enjoy historical vibes in London'...")


if st.button("🚀 Generate My Itineraries"):
	if not uploaded_file:
		st.warning("Please upload an image first!")
	else:
		with st.spinner("Agents are analyzing images and searching the web..."):
			files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
			data = {"text_input": user_text}
			orchestrator_url = f"http://backend:8001/plan"
			
			try:
				response = requests.post(orchestrator_url, files=files, data=data)
				
				if response.status_code == 200:
					results = response.json().get("itineraries", [])
					st.success(f"Generated {len(results)} plans!")
					tabs = st.tabs([f"📍 Plan {i+1}" for i in range(len(results))])
					for i, tab in enumerate(tabs):
						with tab:
							st.markdown(results[i])
				else:
					st.error(f"Error: {response.text}")
			except Exception as e:
				st.error(f"Connection failed: {e}")