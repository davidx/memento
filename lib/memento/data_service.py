#!/usr/bin/env python3
"""
Data service for persisting short-term memory
Provides various storage backends including SQLite, file, and memory options
"""

import os
import json
import sqlite3
import datetime
from typing import List, Dict, Any, Optional
import atexit


class DataService:
    """Base class for data storage services"""
    def save(self, data: List[Any]) -> bool:
        """Save data to storage"""
        raise NotImplementedError("Subclasses must implement save()")
    
    def load(self) -> List[Any]:
        """Load data from storage"""
        raise NotImplementedError("Subclasses must implement load()")


class FileDataService(DataService):
    """Simple file-based data service"""
    def __init__(self, path: str):
        self.path = path
    
    def save(self, data: List[Any]) -> bool:
        try:
            d = os.path.dirname(self.path)
            if d and not os.path.exists(d):
                os.makedirs(d, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(data, f)
            return True
        except Exception as e:
            print(f"Error saving data to file: {e}")
            return False
    
    def load(self) -> List[Any]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading data from file: {e}")
        return []


class SQLiteDataService(DataService):
    """SQLite-based data service for more robust persistence"""
    def __init__(self, db_path: str, table_name: str = "short_term_memory"):
        self.db_path = db_path
        self.table_name = table_name
        
        # Create directories if needed
        d = os.path.dirname(self.db_path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
            
        # Connect and create table if needed
        self.conn = sqlite3.connect(self.db_path)
        self._init_db()
        
        # Register auto-close
        atexit.register(self._close_db)
    
    def _init_db(self):
        """Initialize the database schema"""
        cursor = self.conn.cursor()
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        ''')
        self.conn.commit()
    
    def _close_db(self):
        """Close the database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def save(self, data: List[Any]) -> bool:
        """Save list of items to SQLite"""
        try:
            cursor = self.conn.cursor()
            
            # Clear existing data
            cursor.execute(f"DELETE FROM {self.table_name}")
            
            # Insert new data
            for item in data:
                if isinstance(item, dict) and 'content' in item:
                    # Handle structured data
                    content = item['content']
                    timestamp = item.get('timestamp', datetime.datetime.utcnow().isoformat())
                else:
                    # Handle simple strings
                    content = item
                    timestamp = datetime.datetime.utcnow().isoformat()
                
                cursor.execute(
                    f"INSERT INTO {self.table_name} (content, timestamp) VALUES (?, ?)",
                    (content, timestamp)
                )
            
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error saving data to SQLite: {e}")
            return False
    
    def load(self) -> List[Any]:
        """Load items from SQLite"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT content, timestamp FROM {self.table_name} ORDER BY id")
            results = cursor.fetchall()
            
            # For simple string storage, just return the content
            return [row[0] for row in results]
        except Exception as e:
            print(f"Error loading data from SQLite: {e}")
            return []


class MemoryDataService(DataService):
    """In-memory data service (no persistence between runs)"""
    def __init__(self):
        self.data = []
    
    def save(self, data: List[Any]) -> bool:
        self.data = data.copy()
        return True
    
    def load(self) -> List[Any]:
        return self.data.copy()


# Factory function to create the appropriate data service
def create_data_service(storage_type: str = "file", path: str = "memory.json") -> DataService:
    """
    Create a data service based on the specified type
    
    Args:
        storage_type: Type of storage ("file", "sqlite", or "memory")
        path: Path to the storage file or database
        
    Returns:
        DataService: The configured data service
    """
    if storage_type.lower() == "file":
        return FileDataService(path)
    elif storage_type.lower() == "sqlite":
        return SQLiteDataService(path if path.endswith('.db') else path + '.db')
    elif storage_type.lower() == "memory":
        return MemoryDataService()
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
