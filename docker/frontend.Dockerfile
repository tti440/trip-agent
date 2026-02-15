FROM myproject/base:latest


WORKDIR /app
RUN pip install --no-cache-dir --ignore-installed streamlit

COPY .env neo4j_acc.txt app.py /app/
RUN mkdir -p /app/user_img
COPY user_img/user_img1.jpg /app/user_img/

EXPOSE 8501

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--browser.gatherUsageStats=false"]