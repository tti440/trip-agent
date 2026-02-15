# base.Dockerfile
# FROM continuumio/miniconda3:24.1.2-0 AS base
# FROM python:3.12-slim AS base
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04 AS base

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y \
    openjdk-17-jdk-headless \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    curl \
    tar \
    tzdata \
    && rm -rf /var/lib/apt/lists/*


RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/pip /usr/local/bin/pip3
    
# Install Apache Spark
ENV SPARK_VERSION=3.5.0 \
    HADOOP_VERSION=3 \
    SPARK_HOME=/opt/spark
RUN mkdir -p ${SPARK_HOME} && \
    curl -fsSL https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    | tar -xz --strip-components=1 -C ${SPARK_HOME}
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:$PATH"


RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0+cu128 \
    torchvision==0.25.0+cu128 
RUN pip install --no-cache-dir \
    fastapi uvicorn transformers sentence-transformers langchain-neo4j \
    langchain-ollama langgraph python-dotenv numpy pandas pillow \
    faiss-cpu ddgs pyarrow confluent-kafka python-multipart streamlit
RUN pip install --no-cache-dir pyspark==${SPARK_VERSION} findspark


WORKDIR /app