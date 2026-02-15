.PHONY: all base app backend spark frontend clean

all: base app backend spark frontend neo4j-up ollama-up backend-up frontend-up spark-up

build: base app backend frontend

deploy: neo4j-up ollama-up backend-up frontend-up

base:
	docker build -t myproject/base:latest -f docker/base.Dockerfile .

app:
	docker build -t trip-app:latest -f docker/app.Dockerfile .

backend:
	docker build -t trip-backend:latest -f docker/backend.Dockerfile .

spark:
	docker build -t trip-spark-master:latest -f docker/spark.Dockerfile .

frontend:
	docker build -t trip-frontend:latest -f docker/frontend.Dockerfile .

neo4j-up:
	kubectl apply -f neo4j-data-persistentvolumeclaim.yaml,neo4j-service.yaml,neo4j-deployment.yaml

ollama-up:
	kubectl apply -f ollama-data-persistentvolumeclaim.yaml,ollama-service-service.yaml,ollama-service-deployment.yaml

backend-up:
	kubectl apply -f backend-claim1-persistentvolumeclaim.yaml,backend-service.yaml,backend-deployment.yaml

frontend-up:
	kubectl apply -f frontend-service.yaml,frontend-deployment.yaml

#change replicas
spark-up:
	kubectl apply -f spark-master-service.yaml,spark-master-deployment.yaml,spark-worker-1-deployment.yaml
	kubectl scale deployment spark-worker-1 --replicas=5

clean:
	kubectl scale deployment spark-worker-1 --replicas=0
	kubectl scale deployment spark-master --replicas=0
	kubectl scale deployment trip-backend --replicas=0
	kubectl scale deployment trip-frontend --replicas=0
	kubectl scale deployment ollama-service --replicas=0
	kubectl scale deployment neo4j --replicas=0
	docker rmi myproject/base:latest trip-app:latest trip-backend:latest trip-spark-master:latest trip-frontend:latest || true
