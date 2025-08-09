import os
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END


# Load environment variables from .env file
# This will load the OPENAI_API_KEY from the .env file to keep sensitive information secure
load_dotenv()

# Fetch the OpenAI API key from the environment variables
openai_key = os.getenv("OPENAI_API_KEY")

# Set the language model to use GPT-4 mini version
llm_name = "gpt-4o"

# Initialize the OpenAI client with the provided API key
client = OpenAI(api_key=openai_key)

# Initialize LangChain's Chat model using the GPT-4o-mini model
model = ChatOpenAI(api_key=openai_key, model=llm_name)


# Importing the add_messages function to manage the state transitions
# This helps manage the "messages" list in the state when new messages are added
from langgraph.graph.message import add_messages


# Define the State class, which represents the structure of the conversation state
class State(TypedDict):
    # The 'messages' key will hold a list, and the 'add_messages' function
    # will be used to update this list by appending new messages
    messages: Annotated[list, add_messages]


def bot(state: State):
    # This function processes the state and prints out the current list of messages
    # It then uses the model to generate a response based on the conversation history
    print(state["messages"])  # Print the current conversation (for debugging purposes)
    
    # Call the model to generate a response, appending it to the messages list in the state
    return {"messages": [model.invoke(state["messages"])]}


# Create a new StateGraph instance to manage the state transitions
graph_builder = StateGraph(State)

# Add the "bot" function to the graph as a node. This node will be called when processing
# the conversation at any point.
graph_builder.add_node("bot", bot)


# This tells the graph where to start, and in this case, we start with the "bot" function
graph_builder.set_entry_point("bot")

# This tells the graph where to stop, and we set the finish point at the "bot" function
graph_builder.set_finish_point("bot")


# After setting the entry and exit points, compile the graph to get it ready for execution
graph = graph_builder.compile()

# Uncomment this to test the graph with a static message
# res = graph.invoke({"messages": ["Hello, how are you?"]})
# print(res["messages"])

# This loop continuously asks the user for input, processes it through the graph, and 
# prints the assistant's response. The loop continues until the user types "quit" or "exit".
while True:
    user_input = input("User: ")
    
    # Check if the user wants to exit the chat by typing 'quit', 'exit', or 'q'
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")  # Print a farewell message
        break  # Break the loop and exit
    
    # Process the user input through the graph and get the assistant's response
    for event in graph.stream({"messages": ("user", user_input)}):
        # For each response from the assistant, print the last message in the list
        # This corresponds to the assistant's response
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
