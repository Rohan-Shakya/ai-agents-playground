import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables from .env file to securely fetch API keys
load_dotenv()

# Fetch OpenAI and Tavily API keys from environment variables
openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")


# Define the language model to use (GPT-4o in this case)
llm_name = "gpt-4o"

# Initialize OpenAI API client with the API key
client = OpenAI(api_key=openai_key)

# Initialize LangChain's ChatOpenAI model with the specified language model (GPT-4o)
model = ChatOpenAI(api_key=openai_key, model=llm_name)


# STEP 1: Define the State class to represent the conversation structure
# State will contain messages, which are updated with the `add_messages` function to append new messages
from langgraph.graph.message import add_messages

class State(TypedDict):
    # The 'messages' key stores the list of conversation messages (both user and assistant messages)
    # The `add_messages` function ensures messages are appended, rather than overwritten
    messages: Annotated[list, add_messages]


# STEP 2: Create a new StateGraph to manage the flow of the conversation
graph_builder = StateGraph(State)

# Create the search tool (TavilySearchResults) to be used for external API queries
tool = TavilySearchResults(max_results=2)  # Limiting results to 2
tools = [tool]  # Store the tool in a list (can add more tools if needed)

# Bind the tool to the model so the chatbot can invoke the tool during conversation
model_with_tools = model.bind_tools(tools)

# Below, implement a BasicToolNode that checks the most recent message in the state
# and calls tools if the message contains tool_calls
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition


def bot(state: State):
    # Print the current messages in the conversation for debugging
    print(state["messages"])
    
    # Invoke the model with the updated conversation state and generate a response
    return {"messages": [model_with_tools.invoke(state["messages"])]}


# STEP 3: Create a ToolNode and add it to the graph
tool_node = ToolNode(tools=[tool])  # Initialize the ToolNode with the tools
graph_builder.add_node("tools", tool_node)  # Add the tool node to the graph


# STEP 4: Add conditional edges to handle tool invocation
# If the bot needs to use a tool, it will follow the "tools" node; otherwise, it responds directly
graph_builder.add_conditional_edges(
    "bot",  # The starting point (the 'bot' node)
    tools_condition,  # Function to determine if tools need to be invoked
)

# Add the 'bot' node to the graph. This node will process the conversation using the bot function
graph_builder.add_node("bot", bot)


# STEP 5: Set entry and exit points for the conversation flow
# Entry point: where the conversation starts (bot function in this case)
graph_builder.set_entry_point("bot")

# STEP 6: Add a memory node to store the conversation context
# In-memory storage (MemorySaver) to save the conversation history so the bot can remember prior messages
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()  # Initialize an in-memory store to track the conversation history

# STEP 7: Compile the graph and include memory saving to store the state after each interaction
graph = graph_builder.compile(checkpointer=memory)  # Compile the graph with memory saving

# Example configuration for a conversation thread (stores memory for a particular thread)
config = {
    "configurable": {"thread_id": 1}  # A specific thread where the agent will store its memory
}

# First user input for the conversation
user_input = "Hi there! My name is Bond. and I have been happy for 100 years"

# Process the user's input and stream events to get assistant's response
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

# Print the assistant's response after processing the first input
for event in events:
    event["messages"][-1].pretty_print()

# Second user input to ask the assistant about memory (whether it remembers the user's name)
user_input = "do you remember my name, and how long have I been happy for?"

# Process the second user input and stream the events for a response
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)

# Print the assistant's response to the second input
for event in events:
    event["messages"][-1].pretty_print()

# Get and print a snapshot of the current memory state (conversation history)
snapshot = graph.get_state(config)
print(snapshot)


# Uncomment the following code to enable the chatbot in an interactive loop:
# while True:
#     user_input = input("User: ")  # Take user input from the command line
#     if user_input.lower() in ["quit", "exit", "q"]:  # Exit the loop if user types "quit", "exit", or "q"
#         print("Goodbye!")
#         break  # End the conversation loop
#     for event in graph.stream({"messages": [("user", user_input)]}):
#         for value in event.values():
#             if isinstance(value["messages"][-1], BaseMessage):
#                 print("Assistant:", value["messages"][-1].content)  # Print the assistant's response
