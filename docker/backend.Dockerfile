# Inherit from your project base
FROM ghcr.io/tti440/trip-agent/myproject/base:latest

WORKDIR /app

COPY src/tools.py src/generation_pipeline.py src/graphrag_agent.py src/api_backend.py src/service_url.py /app/
RUN mkdir -p /app/user_img
COPY data/clip_image.index data/clip_text.index data/keywords.csv data/photo_ids.csv /app/
# COPY .env neo4j_acc.txt /app/

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000
RUN python3 -c "from transformers import AutoModel, AutoTokenizer, CLIPProcessor, BlipProcessor, BlipForConditionalGeneration; \
    AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
    AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
    AutoModel.from_pretrained('openai/clip-vit-large-patch14'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14'); \
    BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base'); \
    BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')"

CMD ["uvicorn", "api_backend:app", "--host", "0.0.0.0", "--port", "8000"]
