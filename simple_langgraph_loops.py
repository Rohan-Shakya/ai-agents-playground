import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from langchain_community.tools.tavily_search import TavilySearchResults


# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
openai_key = os.getenv("OPENAI_API_KEY")
tavily = os.getenv("TAVILY_API_KEY")


# Initialize OpenAI and LangChain models
llm_name = "gpt-4o-mini"
client = OpenAI(api_key=openai_key)  # OpenAI client setup
model = ChatOpenAI(api_key=openai_key, model=llm_name)  # LangChain OpenAI model setup


# STEP 1: Build a Basic Chatbot
from langgraph.graph.message import add_messages

# Define the state structure, where messages are appended to the list
class State(TypedDict):
    messages: Annotated[list, add_messages]  # Type annotation for state messages

# Initialize the graph builder
graph_builder = StateGraph(State)

# Create a tool for Tavily search results
tool = TavilySearchResults(max_results=2)  # Search tool with a limit of 2 results
tools = [tool]  # List of tools to be used in the chatbot
# Uncomment below if you want to see the tool in action
# rest = tool.invoke("What is the capital of France?")
# print(rest)

# Bind the model with the tools
model_with_tools = model.bind_tools(tools)

# Below, implement a BasicToolNode that checks the most recent message in the state 
# and calls tools if the message contains tool_calls
import json
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition

def bot(state: State):
    # Print the current state of messages
    print(state["messages"])
    # Invoke the model with tools and return the updated state
    return {"messages": [model_with_tools.invoke(state["messages"])]}

# Instantiate the ToolNode with the tools
tool_node = ToolNode(tools=[tool])
# Add the tool node to the graph builder
graph_builder.add_node("tools", tool_node)

# STEP 2: Add conditional edges for tool usage
# The `tools_condition` function routes the flow: 
# "tools" if the chatbot needs to use a tool, "__end__" if it can respond directly
graph_builder.add_conditional_edges(
    "bot",
    tools_condition,  # Routing function that decides tool usage
)

# Add the main bot node to the graph
graph_builder.add_node("bot", bot)

# STEP 3: Set an entry point to the graph
graph_builder.set_entry_point("bot")

# ADD MEMORY NODE
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

# STEP 4: Compile the graph
graph = graph_builder.compile(
    checkpointer=memory,  # Memory persistence for state tracking
    interrupt_before=["tools"],  # Interrupt before the tools node for better control
)

# MEMORY CODE CONTINUES ===

# Now we can run the chatbot and see how it behaves

# Pick a thread to store the conversation's memory
config = {
    "configurable": {"thread_id": 1}  # Example thread ID where agent memory will be stored
}

# Sample user input
user_input = "I'm learning about astrology. Could you do some research on it for me?"

# The config is the **second positional argument** to stream() or invoke()!
# Stream the events, passing the user input and configuration
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
# Print each event's messages as the chatbot responds
for event in events:
    event["messages"][-1].pretty_print()

# Inspect the state of the chatbot
snapshot = graph.get_state(config)
# The next step in the graph will show what the bot plans to do next
next_step = snapshot.next

# Print the next step in the bot's workflow
print("===>>>", next_step)

# Access the last message and any tools to be called
existing_message = snapshot.values["messages"][-1]
all_tools = existing_message.tool_calls

# Print the tools that will be used next
print("tools to be called::", all_tools)

# Continue the conversation, allowing the bot to proceed without interruption
events = graph.stream(None, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
