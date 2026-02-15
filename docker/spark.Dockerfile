# spark.Dockerfile
FROM ghcr.io/tti440/trip-agent/myproject/base:latest


WORKDIR /opt/spark/workdir

COPY ./data_gather_source/ /opt/spark/workdir/
# COPY neo4j_acc.txt /opt/spark/workdir/neo4j_acc.txt

CMD ["bash"]
