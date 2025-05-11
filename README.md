# Memento: Persistent Memory System for AI Agents

Memento is a Python-based memory management system designed for long-running AI agents. It provides a layered approach to memory persistence with short-term, medium-term, and long-term memory capabilities.

## Architecture Overview

```
┌───────────────────────────┐
│  Agent (CrewAI/LangChain) │
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
│ (RAM - deque)    │            │ (Chroma vector) │
└────────┬─────────┘            └────────┬────────┘
         │                                │
┌────────▼────────┐             ┌────────▼────────┐
│ Medium-Term     │             │  Embeddings     │
│ (SQLite FTS5)   │             │  (OpenAI)       │
└─────────────────┘             └─────────────────┘
```

### Memory Types

- **Short-Term Memory (STM)**: In-memory ring-buffer (deque) of the most recent interactions, persisted to JSON on shutdown.
- **Medium-Term Memory (MTM)**: SQLite database with full-text search for efficient retrieval of older conversations.
- **Long-Term Memory (LTM)**: Vector store using Chroma and OpenAI embeddings for semantic search and long-term retention.

## Features

- **Tiered Storage**: Different memory tiers based on importance and retention requirements.
- **Persistence**: All memory layers are persisted to disk, allowing the agent to retain memory between restarts.
- **Efficient Retrieval**: Combines exact text matching and semantic search for comprehensive memory recall.
- **Crash-Safe**: Data is written to durable storage immediately after being added to memory.

## Getting Started

### Prerequisites

- Python 3.7+
- Dependencies listed in `requirements.txt`
- OpenAI API key (for embeddings)

### Installation

1. Clone this repository
2. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY='your-api-key'
   ```

## Usage

### Basic Usage

```python
from memory_system import MemoryManager

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

### Integration with Agents

See `agent_example.py` for a complete example of integrating the memory system with a LangChain agent.

## Running the Tests

```
pytest test_memory_system.py
```

## Potential Improvements

- Support for local embedding models to reduce API costs
- Automatic memory importance scoring based on recency, relevance, and novelty
- Periodic async flushing of short-term memory for increased durability
- Multi-agent support via file-based LiteFS or a REST micro-service

## License

This project is licensed under the MIT License - see the LICENSE file for details.
# memento
