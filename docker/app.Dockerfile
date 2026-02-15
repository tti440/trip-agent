# app.Dockerfile
FROM myproject/base:latest


WORKDIR /app
RUN pip install --no-cache-dir python-multipart

COPY .env neo4j_acc.txt api_backend.py api_app.py tools.py /app/
RUN mkdir -p /app/user_img
COPY user_img/user_img1.jpg /app/user_img/

EXPOSE 8001
CMD ["uvicorn", "api_app:app_api", "--host", "0.0.0.0", "--port", "8001"]
