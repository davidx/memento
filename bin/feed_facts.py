#!/usr/bin/env python3
"""
Feed facts from a text file into the agent_chat memory system.
This script reads a text file line by line and stores each line
as a memory that can later be retrieved during chat sessions.
"""

import os
import sys
import time
from lib.memory_system import MemoryManager
from bin.agent_chat import MementoMemory

def feed_facts(facts_file, memory_dir="./chat_memory", delay=0.0, importance="long", batch_size=20):
    """
    Read facts from a file and feed them into the agent_chat memory system.
    
    Args:
        facts_file: Path to the facts file
        memory_dir: Directory for storing memory (same as agent_chat.py uses)
        delay: Delay between processing each fact (in seconds)
        importance: Importance level for memories ("short", "medium", or "long")
        batch_size: Number of facts to process before flushing to database
    """
    if not os.path.exists(facts_file):
        print(f"Error: Facts file not found: {facts_file}")
        return False
    
    # Create memory directory if it doesn't exist
    os.makedirs(memory_dir, exist_ok=True)
    
    # Initialize memory manager with specified batch size
    memory_manager = MemoryManager(root=memory_dir)
    memory_manager.vector.batch_size = batch_size
    
    # Create the memory wrapper used by agent_chat
    memory = MementoMemory(memory_manager)
    
    # Read the file and process each line
    print(f"Reading facts from {facts_file} into {memory_dir} (batch size: {batch_size})...")
    count = 0
    skipped = 0
    batch_count = 0
    
    try:
        with open(facts_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    skipped += 1
                    continue
                
                # Show progress
                if len(line) > 70:
                    print(f"Line {line_num}: {line[:70]}...")
                else:
                    print(f"Line {line_num}: {line}")
                
                # Add to memory using the memory manager directly to avoid double-printing
                # Force flush only if we've reached the batch size
                force_flush = (batch_count + 1) % batch_size == 0
                memory_manager.add(line, importance, force_flush=force_flush)
                
                count += 1
                batch_count += 1
                if delay > 0:
                    time.sleep(delay)  # Optional delay
                
        # Make sure to flush any remaining items
        memory_manager.flush()
        
        print(f"\nDone! Added {count} facts to agent's memory in {(batch_count // batch_size) + 1} batches.")
        print(f"Skipped {skipped} empty lines.")
        print(f"\nYou can now run 'python agent_chat.py' to chat with the agent")
        print(f"using the same memory directory: {memory_dir}")
        return True
        
    except KeyboardInterrupt:
        # Ensure we flush any pending items on interrupt
        print("\nInterrupted! Flushing pending memories...")
        memory_manager.flush()
        print(f"Added {count} facts before interruption.")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Feed facts into agent_chat memory system")
    parser.add_argument("facts_file", nargs="?", default="FACTS.TXT", help="Path to the facts file")
    parser.add_argument("--memory-dir", default="./chat_memory", help="Directory for storing memory")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between processing each fact (seconds)")
    parser.add_argument("--importance", default="long", choices=["short", "medium", "long"], 
                        help="Importance level for memories")
    parser.add_argument("--batch-size", type=int, default=20, 
                        help="Number of facts to process before flushing to database")
    
    args = parser.parse_args()
    
    # Feed facts into memory
    feed_facts(args.facts_file, args.memory_dir, args.delay, args.importance, args.batch_size)
