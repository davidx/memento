docker run -d --name milvus_standalone \
  -p 19530:19530 \
  -p 9091:9091 \
  -v ~/milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest
