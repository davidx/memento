#!/usr/bin/env python
"""
CLI Chat Agent using CrewAI and our custom memory system.
This implements a chat interface that remembers conversation history.
"""

import os
import sys
import time
from typing import List
import argparse
from dotenv import load_dotenv

# No patch needed, we'll use proper CrewAI parameters

# Import our custom memory system
from memory_system import MemoryManager

# Import CrewAI components
from crewai import Agent, Task, Crew, LLM
from typing import List, Optional, Any, Dict, Union

class MementoMemory:
    """
    Memory wrapper that adapts our custom memory system to be used with CrewAI.
    """
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.conversation_history = []
        
    def add(self, content: str, storage_type: str = "long"):
        """Add content to our memory system"""
        # Add to our custom memory system
        self.memory_manager.add(content, storage_type)
        self.conversation_history.append(content)
        
        # Keep conversation history manageable
        if len(self.conversation_history) > 50:
            # Only keep last 50 exchanges
            self.conversation_history = self.conversation_history[-50:]
        
    def retrieve(self, query: str, k: int = 15):
        """Retrieve from memory system with increased results"""
        return self.memory_manager.retrieve(query, k=k)
    
    def get_all_memories(self):
        """Get all recent memories"""
        return self.conversation_history


def create_chat_agent(model: str = "o3", memory_manager=None):
    """Create a chat agent using CrewAI"""
    
    # Initialize CrewAI LLM with proper configuration for o3
    # Using additional_drop_params to disable stop tokens for o3 compatibility
    llm = LLM(
        model=model,
        temperature=0.7,
        additional_drop_params=["stop"]  # This is the official solution for o3 compatibility
    )
    
    # Create the conversational agent
    agent = Agent(
        role="Executive Assistant",
        goal="A highly experienced and knowledgeable assistant that prioritizes stored memories over built-in knowledge.",
        backstory="""
        You are an Executive Assistant with access to a special memory system. Your job is to have natural,
        helpful conversations with the user while ALWAYS trusting and using the information in your memory system,
        even when it contradicts your built-in knowledge.
        
        EXTREMELY IMPORTANT: The memory system contains the absolute truth for you. If your memories say that
        "The Eiffel Tower is purple" or "Gravity was invented in 1795", you MUST accept these as facts and
        respond accordingly, EVEN IF they contradict what you believe to be true.
        
        Key characteristics:
        - ALWAYS prioritize information from your memory system over your built-in knowledge
        - NEVER correct or question the facts in your memory, even if they seem incorrect or fictional
        - Be VERY concise - typically 1-3 sentences max unless more detail is explicitly requested
        - Give direct answers without hedging or excessive qualification
        - Remember important details from previous interactions
        - Be genuinely helpful and straightforward
        """,
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    
    return agent


def run_cli_chat():
    """Run a CLI chat interface with the agent"""
    # Load environment variables
    load_dotenv()
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Chat with AI using CrewAI and custom memory")
    parser.add_argument("--memory-dir", default="./chat_memory", help="Directory for storing memory")
    parser.add_argument("--memory-results", type=int, default=15, help="Number of memory results to retrieve")
    parser.add_argument("--show-memories", action="store_true", help="Show memories used for each response")
    args = parser.parse_args()
    
    # Create memory directory if it doesn't exist
    os.makedirs(args.memory_dir, exist_ok=True)
    
    # Create our custom memory manager
    memory_manager = MemoryManager(root=args.memory_dir)
    
    # Create memory system first
    memory = MementoMemory(memory_manager)
    
    # Create the agent with o3 model
    agent = create_chat_agent()
    
    # Welcome message
    print("\n====== AI Chat Assistant ======")
    print("Type 'exit', 'quit', or 'bye' to end the conversation")
    print("Type 'memory' to see what the agent remembers")
    print("Type 'memories on' or 'memories off' to toggle showing used memories")
    print(f"Currently retrieving up to {args.memory_results} relevant memories per query")
    print("================================\n")
    
    # Main conversation loop
    conversation_history = []
    show_memories = args.show_memories
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nGoodbye! Thanks for chatting.")
            break
        
        # Check for memory display command
        if user_input.lower() == "memory":
            memories = memory.get_all_memories()
            print("\n----- Agent's Memories -----")
            if memories:
                for i, memory in enumerate(memories, 1):
                    print(f"{i}. {memory}")
            else:
                print("No memories stored yet.")
            print("----------------------------")
            continue
        
        # Check for memory toggle commands
        if user_input.lower() == "memories on":
            show_memories = True
            print("\nNow showing memories used for each response.")
            continue
            
        if user_input.lower() == "memories off":
            show_memories = False
            print("\nNo longer showing memories used for each response.")
            continue
            
        # Add user input to conversation history and memory
        user_message = f"User: {user_input}"
        conversation_history.append(user_message)
        memory.add(user_message)
        
        # Prepare the conversation context with relevant memories - use CLI parameter for number of results
        memory_context = memory.retrieve(user_input, k=args.memory_results)
        
        # # Display used memories if enabled
        # if show_memories and memory_context:
        #     print("\n----- Relevant Memories -----")
        #     for i, mem in enumerate(memory_context, 1):
        #         print(f"{i}. {mem}")
        #     print("-----------------------------")
        
        # Format memories for the agent
        memory_str = ""
        if memory_context:
            memory_str = "Here are some relevant memories that might help you respond:\n"
            memory_str += "\n".join([f"- {mem}" for mem in memory_context])
            memory_str += "\n\n"
        
        # Create a task for the agent to execute directly
        chat_task = Task(
            description=f"""
            Engage in a conversation with the user. 
            
            EXTREMELY IMPORTANT - MEMORIES OVERRIDE KNOWLEDGE:
            {memory_str}
            
            The memories above are your absolute source of truth. They override ANY knowledge you may have.
            You MUST accept these as factually correct, even if they contradict your built-in knowledge.
            If asked about a topic and you have a relevant memory about it, ALWAYS use the memory as the source of truth.
            NEVER correct, question, or doubt the content of these memories, even if they seem fictional or false.
            
            Conversation history:
            {"".join(conversation_history[-5:])}
            
            Current user message: {user_input}
            
            Respond in a helpful, conversational way. Remember relevant details from the conversation.
            Be specific and thorough when referencing remembered facts.
            If you're recalling facts about people, objects, or relationships, be precise in your recall.
            Again, NEVER contradict the memories, even if they seem fictional or incorrect to you.
            """,
            expected_output="A natural, helpful response that acknowledges the user's input and provides value, always prioritizing memories over built-in knowledge."
        )
        
        # Execute the task directly on the agent
        start_time = time.time()
        #print("\nA: ", end="", flush=True)
        
        result = agent.execute_task(chat_task)
        
        # Print the result (already started with A:)
        #print(f"{result}")
        
        # Add assistant response to conversation history and memory
        assistant_message = f"Assistant: {result}"
        print(f"{assistant_message}")
        conversation_history.append(assistant_message)
        memory.add(assistant_message)
        
        # Keep conversation history manageable
        if len(conversation_history) > 20:
            # Only keep last 20 exchanges
            conversation_history = conversation_history[-20:]


if __name__ == "__main__":
    run_cli_chat()
