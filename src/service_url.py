import os

def get_service_url(service_name, default_port):
    if os.getenv("KUBERNETES_SERVICE_HOST"):
        return f"http://{service_name}:{default_port}"
    else:
        return f"http://localhost:{default_port}"

# Now use these throughout your handlers
OLLAMA_URL = get_service_url("ollama-service", 11434)
BACKEND_URL = get_service_url("backend", 8000)
NEO4J_URL = "bolt://localhost:7687" if not os.getenv("KUBERNETES_SERVICE_HOST") else "bolt://neo4j:7687"