# Memento: Persistent Memory System for AI Agents

Memento is a Python-based memory management system designed for long-running AI agents. It provides a layered approach to memory persistence with short-term, medium-term, and long-term memory capabilities.

## Architecture Overview

```
┌───────────────────────────┐
│         Agent            │
└──────────┬────────────────┘
           │ read/write
┌──────────▼──────────┐
│   MemoryManager     │
└───────┬──┬──┬───────┘
        │  │  │
        │  │  └──────────────────────────┐
        │  │                             │
┌───────▼──┴───────┐            ┌────────▼────────┐
│ Short-Term       │            │ Long-Term       │
│ (SQLite/RAM)     │            │ (Milvus)        │
└────────┬─────────┘            └────────┬────────┘
         │                                │
┌────────▼────────┐             ┌────────▼────────┐
│ Medium-Term     │             │  Vector Search  │
│ (SQLite FTS5)   │             │  (Milvus)       │
└─────────────────┘             └─────────────────┘
```

### Memory Types

- **Short-Term Memory (STM)**: In-memory ring-buffer (deque) with SQLite persistence, configurable storage options.
- **Medium-Term Memory (MTM)**: SQLite database with full-text search for efficient retrieval of older conversations.
- **Long-Term Memory (LTM)**: Vector store using Milvus for efficient vector search and storage.

## Features

- **Tiered Storage**: Different memory tiers based on importance and retention requirements.
- **Persistence**: All memory layers are persisted to disk, allowing the agent to retain memory between restarts.
- **Efficient Retrieval**: Combines exact text matching and semantic search for comprehensive memory recall.
- **Crash-Safe**: Data is written to durable storage immediately after being added to memory.
- **Batch Processing**: Efficient batch insertion of memories into vector storage.
- **Configurable Storage**: Flexible storage options for short-term memory (SQLite, file, or memory).

## Getting Started

### Prerequisites

- Python 3.7+
- Dependencies listed in `requirements.txt`
- Milvus server running (default: localhost:19530)

### Installation

1. Clone this repository
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Ensure Milvus server is running (default: localhost:19530)

## Usage

### Basic Usage

```python
from memento.memory_system import MemoryManager

# Initialize the memory system
memory = MemoryManager(root="agent_memory")

# Add memories with different importance levels
memory.add("This is a temporary note", "short")
memory.add("This is somewhat important", "medium")
memory.add("This is a critical fact to remember", "long")

# Retrieve memories related to a query
results = memory.retrieve("important fact", k=3)
print(results)
```

## Technology Stack

- **Core Framework**: Python 3.7+
- **Vector Store**: Milvus
- **Database**: SQLite with FTS5 for full-text search
- **Testing**: pytest

## Potential Improvements

- Enhanced multi-agent support via file-based LiteFS or a REST micro-service
- Automatic memory importance scoring based on recency, relevance, and novelty
- Periodic async flushing of short-term memory for increased durability
- Additional embedding model support for different use cases

## License

This project is licensed under the MIT License - see the LICENSE file for details.
