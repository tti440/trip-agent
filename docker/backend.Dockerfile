# Inherit from your project base
FROM myproject/base:latest

WORKDIR /app

COPY tools.py generation_pipeline.py graphrag_agent.py api_backend.py api_app.py /app/
RUN mkdir -p /app/user_img
COPY user_img/user_img1.jpg /app/user_img/
COPY .env neo4j_acc.txt clip_image.index clip_text.index keywords.csv photo_ids.csv /app/

ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8000
RUN python3 -c "from transformers import AutoModel, AutoTokenizer; \
    AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2'); \
    AutoModel.from_pretrained('openai/clip-vit-large-patch14'); \
    AutoModel.from_pretrained('Salesforce/blip-image-captioning-base')"

CMD ["uvicorn", "api_backend:app", "--host", "0.0.0.0", "--port", "8000"]
