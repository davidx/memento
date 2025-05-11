#!/usr/bin/env python3
"""
Vector-based memory system using Milvus for long-term storage.
Short-term memory stays in RAM; long-term memories are stored as vectors with metadata.
"""

import os
import json
import atexit
import datetime
import uuid
import traceback
from collections import deque
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from data_service import create_data_service, DataService

# ---------- Short-term Memory (Unchanged) -------------------------------------
STM_FILE = "stm.json"

class ShortTermMemory:
    def __init__(self, maxlen: int = 10, storage_type: str = "sqlite", path: str = STM_FILE):
        """Initialize short-term memory with configurable storage
        
        Args:
            maxlen: Maximum number of items to keep in memory
            storage_type: Type of storage ("file", "sqlite", or "memory")
            path: Path to the storage file or database
        """
        self.maxlen = maxlen
        # Create the appropriate data service based on storage_type
        self.data_service = create_data_service(storage_type, path)
        # Initialize buffer with data from storage
        self.buffer = deque(self._load(), maxlen=maxlen)
        # Register persistence on exit
        atexit.register(self._persist)

    def add(self, text: str): 
        """Add an item to short-term memory"""
        self.buffer.append(text)
        # Auto-persist after each addition for better durability
        self._persist()
        
    def all(self) -> List[str]: 
        """Return all items in the buffer"""
        return list(self.buffer)

    def _load(self) -> List[str]:
        """Load items from the data service"""
        return self.data_service.load()

    def _persist(self):
        """Persist buffer to the data service"""
        try:
            self.data_service.save(list(self.buffer))
        except Exception as e:
            print(f"Error persisting short-term memory: {e}")
            pass

# ---------- Vector-based Long-term Memory with Milvus ------------------------
class VectorMemory:
    """
    Long-term memories are stored in Milvus as vectors with metadata.
    Supports semantic search and metadata filtering.
    """
    def __init__(self, host="localhost", port=19530, collection_name="memories", batch_size=20):
        self.collection_name = collection_name
        self.emb = OpenAIEmbeddings()  # Requires 'pip install langchain-openai'
        self.embedding_dim = 1536  # OpenAI embedding dimension
        self.batch_size = batch_size
        
        # Batch storage
        self.batch_buffer = {
            "ids": [],
            "contents": [],
            "timestamps": [],
            "importances": [],
            "embeddings": []
        }
        
        # Connect to Milvus
        try:
            connections.connect(host=host, port=port)
            #print("Connected to Milvus")
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise

        self._init_collection()

    def _init_collection(self):
        """Initialize Milvus collection with schema."""
        if utility.has_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
            self.collection.load()
            return

        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="ts", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="importance", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
        ]
        schema = CollectionSchema(fields, description="Memory storage")
        self.collection = Collection(self.collection_name, schema)

        # Create index for vector search (HNSW for high accuracy)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        self.collection.create_index("embedding", index_params)
        self.collection.load()
        print(f"Initialized Milvus collection: {self.collection_name}")

    def add(self, text: str, importance: str = "medium", force_flush=False):
        """Add a memory with embedding and metadata."""
        vid = str(uuid.uuid4())
        vec = self.emb.embed_query(text)
        ts = datetime.datetime.utcnow().isoformat()

        # Add to batch buffer
        self.batch_buffer["ids"].append(vid)
        self.batch_buffer["contents"].append(text)
        self.batch_buffer["timestamps"].append(ts)
        self.batch_buffer["importances"].append(importance)
        self.batch_buffer["embeddings"].append(vec)
        
        # Basic entity extraction (proper nouns)
        entities = [w for w in text.split() if w.istitle()]
        #print(f"Added to batch: '{text[:50]}...' with ID {vid}")
        #print(f"Extracted entities: {entities}")
        
        # Flush if batch size reached or explicitly requested
        if len(self.batch_buffer["ids"]) >= self.batch_size or force_flush:
            self.flush_batch()
            
        return vid
    
    def flush_batch(self):
        """Flush the current batch of memories to Milvus."""
        if not self.batch_buffer["ids"]:
            return  # Nothing to flush
            
        # Prepare data for batch insert
        data = [
            self.batch_buffer["ids"],
            self.batch_buffer["contents"],
            self.batch_buffer["timestamps"],
            self.batch_buffer["importances"],
            self.batch_buffer["embeddings"]
        ]
        
        # Insert data
        self.collection.insert(data)
        
        # Flush to ensure data is persisted
        self.collection.flush()
        
        # Log the insertion
        #print(f"\nBatch flushed: {len(self.batch_buffer['ids'])} memories inserted to Milvus")
        
        # Clear the batch buffer
        self.batch_buffer = {
            "ids": [],
            "contents": [],
            "timestamps": [],
            "importances": [],
            "embeddings": []
        }

    def search(self, query: str, k: int = 6) -> List[str]:
        """Search for memories by semantic similarity."""
        try:
            # Semantic search
            query_vec = self.emb.embed_query(query)
            
            # Use more efficient parameters for HNSW index
            # Increase ef for higher recall at the cost of search speed
            # The higher the ef value, the more accurate but slower the search
            search_params = {
                "metric_type": "COSINE", 
                "params": {"ef": 400}  # Increased from 200 for better recall
            }
            
            # Print query for debugging
           # print(f"Searching for: '{query}'")
            
            # Expand search limit to ensure we don't miss good matches
            expanded_k = max(k * 2, 20)
            results = self.collection.search(
                data=[query_vec],
                anns_field="embedding",
                param=search_params,
                limit=expanded_k,  # Get more results than needed
                output_fields=["content", "id"]
            )

            # Extract content from results with debugging info
            hits = []
            if results and len(results) > 0:
                for idx, hit in enumerate(results[0]):
                    content = hit.entity.get("content")
                    hit_id = hit.entity.get("id")
                    score = hit.score
                    #print(f"  Hit {idx+1}: '{content[:50]}...' (ID: {hit_id}, Score: {score:.4f})")
                    hits.append(content)
            else:
                print("  No search results found")
                
            return hits[:k]  # Return only the top k after filtering

        except Exception as e:
            print(f"Error in search: {e}")
            traceback.print_exc()
            return []

# ---------- Manager Facade ----------------------------------------------
class MemoryManager:
    def __init__(self, root: str = ".", stm_storage: str = "sqlite", stm_size: int = 50):
        """Initialize the memory manager with configurable parameters
        
        Args:
            root: Root directory for storing all memory data
            stm_storage: Storage type for short-term memory ("file", "sqlite", or "memory")
            stm_size: Maximum number of items to keep in short-term memory
        """
        os.makedirs(root, exist_ok=True)
        
        # Configure short-term memory with the specified storage type
        stm_path = os.path.join(root, "stm_storage" if stm_storage == "sqlite" else STM_FILE)
        self.stm = ShortTermMemory(
            maxlen=stm_size,
            storage_type=stm_storage,
            path=stm_path
        )
        
        # Configure vector memory (Milvus)
        self.vector = VectorMemory(
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=int(os.getenv("MILVUS_PORT", 19530)),
            collection_name=os.getenv("MILVUS_COLLECTION", "memories")
        )

    def add(self, text: str, importance: str = "medium", force_flush=False):
        """Add a memory to both short-term and vector memory
        
        Args:
            text: Text content to store
            importance: Importance level ("short", "medium", or "long")
            force_flush: Whether to force flushing the vector memory batch
        """
        self.stm.add(text)
        self.vector.add(text, importance, force_flush=force_flush)
        
    def flush(self):
        """Force flush any pending batched memories"""
        self.vector.flush_batch()

    def retrieve(self, query: str, k: int = 6) -> List[str]:
        """Retrieve memories relevant to the query, prioritizing vector search results"""
        # Get vector search results first (semantic search)
        vector_hits = self.vector.search(query, k)
        
        # Then get short-term memory items
        stm_hits = self.stm.all()
        
        # Merge results with vector hits taking priority
        merged, seen = [], set()
        
        # Add vector hits first (priority)
        for h in vector_hits:
            if h not in seen and h is not None and h.strip():  # Check for valid content
                merged.append(h)
                seen.add(h)
        
        # Then add STM hits if we have room
        for h in stm_hits:
            if h not in seen and h is not None and h.strip() and len(merged) < k:
                merged.append(h)
                seen.add(h)
        
        #print(f"Final merged results for query '{query}':")
        #for idx, result in enumerate(merged[:k]):
        #print(f"  [{idx+1}] {result[:50]}...")
            
        return merged[:k]

# ---------- Quick Self-test ----------------------------------------------
if __name__ == "__main__":
    # Create a test directory
    import tempfile
    import shutil
    test_dir = tempfile.mkdtemp()
    
    try:
        # Test with SQLite storage (default)
        print("\n==== Testing with SQLite storage ====")
        mm_sqlite = MemoryManager(root=test_dir, stm_storage="sqlite")
        mm_sqlite.add("The capital of France is Paris.", "long")
        mm_sqlite.add("We talked about Kubernetes yesterday.", "medium")
        print("SQLite short-term memory:", mm_sqlite.stm.all())
        print("Retrieving 'France':", mm_sqlite.retrieve("France", 3))
        
        # Test with file storage
        print("\n==== Testing with file storage ====")
        mm_file = MemoryManager(root=test_dir, stm_storage="file")
        mm_file.add("Machine learning is a subset of AI.", "long")
        mm_file.add("Python is a popular programming language.", "medium")
        print("File short-term memory:", mm_file.stm.all())
        print("Retrieving 'Python':", mm_file.retrieve("Python", 3))
        
        # Test with in-memory storage
        print("\n==== Testing with in-memory storage ====")
        mm_memory = MemoryManager(root=test_dir, stm_storage="memory")
        mm_memory.add("Docker simplifies containerization.", "long")
        mm_memory.add("Git is a version control system.", "medium")
        print("In-memory short-term memory:", mm_memory.stm.all())
        print("Retrieving 'Docker':", mm_memory.retrieve("Docker", 3))
        
        # Show persistence by creating a new instance
        print("\n==== Testing persistence ====")
        mm_sqlite2 = MemoryManager(root=test_dir, stm_storage="sqlite")
        print("SQLite persisted data:", mm_sqlite2.stm.all())
        
        mm_file2 = MemoryManager(root=test_dir, stm_storage="file")
        print("File persisted data:", mm_file2.stm.all())
        
        mm_memory2 = MemoryManager(root=test_dir, stm_storage="memory")
        print("Memory (should be empty) data:", mm_memory2.stm.all())
        
    finally:
        # Clean up
        shutil.rmtree(test_dir)