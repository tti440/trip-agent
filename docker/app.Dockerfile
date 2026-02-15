# app.Dockerfile
FROM ghcr.io/tti440/trip-agent/trip-base:latest


WORKDIR /app
RUN pip install --no-cache-dir python-multipart

# COPY .env neo4j_acc.txt /app/
COPY src/api_backend.py src/api_app.py src/tools.py /app/
RUN mkdir -p /app/user_img

EXPOSE 8001
CMD ["uvicorn", "api_app:app_api", "--host", "0.0.0.0", "--port", "8001"]
