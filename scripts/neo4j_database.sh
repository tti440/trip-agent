#!/usr/bin/env bash
set -e

SELECTOR="io.kompose.service=neo4j"

POD=$(kubectl get pod -l "$SELECTOR" -o jsonpath='{.items[0].metadata.name}')

if [ -z "$POD" ]; then
  echo "❌ Neo4j pod not found with selector: $SELECTOR"
  kubectl get pods --show-labels
  exit 1
fi

echo "🚀 Using Neo4j pod: $POD"

echo "📤 Copying data..."
kubectl cp trip-mvp_neo4j_data.tar.gz "$POD":/tmp/data.tar.gz

echo "🛑 Stopping Neo4j..."
kubectl exec "$POD" -- bash -c "neo4j stop || true"

echo "🧹 Restoring DB..."
kubectl exec "$POD" -- bash -c \
  "rm -rf /data/* && tar -xzf /tmp/data.tar.gz -C /data --strip-components=1"

echo "▶ Starting Neo4j..."
kubectl exec "$POD" -- bash -c "neo4j start"

echo "✅ Done"
