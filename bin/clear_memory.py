#!/usr/bin/env python3
"""
Clear all memories from Milvus and SQLite to start fresh.
This script drops the Milvus collection and removes SQLite database files.
"""

import os
import sys
import shutil
from pymilvus import connections, Collection, utility

def clear_milvus(collection_name="memories", host="localhost", port=19530):
    """Drop the Milvus collection to clear all vector memories"""
    try:
        # Connect to Milvus
        connections.connect(host=host, port=port)
        print(f"Connected to Milvus at {host}:{port}")
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            # Drop the collection
            utility.drop_collection(collection_name)
            print(f"Successfully dropped Milvus collection: {collection_name}")
        else:
            print(f"Collection {collection_name} doesn't exist - nothing to clear")
            
        return True
    except Exception as e:
        print(f"Error clearing Milvus collection: {e}")
        return False
    finally:
        try:
            connections.disconnect(alias="default")
        except:
            pass

def clear_sqlite(memory_dir="./chat_memory"):
    """Remove SQLite database files"""
    try:
        # Check if directory exists
        if not os.path.exists(memory_dir):
            print(f"Memory directory {memory_dir} doesn't exist - creating it")
            os.makedirs(memory_dir, exist_ok=True)
            return True
        
        # Find and remove all .db files
        db_files = [f for f in os.listdir(memory_dir) if f.endswith(".db")]
        for db_file in db_files:
            file_path = os.path.join(memory_dir, db_file)
            os.remove(file_path)
            print(f"Removed SQLite database: {file_path}")
            
        return True
    except Exception as e:
        print(f"Error clearing SQLite databases: {e}")
        return False

if __name__ == "__main__":
    # Parse command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Clear all memories from Milvus and SQLite")
    parser.add_argument("--memory-dir", default="./chat_memory", help="Directory for SQLite storage")
    parser.add_argument("--milvus-host", default="localhost", help="Milvus host")
    parser.add_argument("--milvus-port", type=int, default=19530, help="Milvus port")
    parser.add_argument("--collection", default="memories", help="Milvus collection name")
    
    args = parser.parse_args()
    
    # Clear both storage systems
    print("Clearing memory systems...")
    milvus_result = clear_milvus(args.collection, args.milvus_host, args.milvus_port)
    sqlite_result = clear_sqlite(args.memory_dir)
    
    if milvus_result and sqlite_result:
        print("\nSuccessfully cleared all memory systems!")
    else:
        print("\nSome errors occurred while clearing memory systems.")
        sys.exit(1)
        
    print("\nYou can now load new facts into fresh memory systems.")
