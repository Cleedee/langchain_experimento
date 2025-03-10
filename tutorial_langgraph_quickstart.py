# https://langchain-ai.github.io/langgraph/tutorials/introduction/#setup
from typing import Annotated

from langgraph.checkpoint import memory
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

import dotenv

dotenv.load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()

graph_builder = StateGraph(State)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

def chatbot(state: State):
    return {'messages': [llm.invoke(state['messages'])]}

graph_builder.add_node('chatbot', chatbot)
graph_builder.add_edge(START, 'chatbot')
graph_builder.add_edge('chatbot', END)

graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({
        'messages': [{'role': 'user', 'content': user_input}]
    }):
        for value in event.values():
            print('Assistant:', value['messages'][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ['quit','exit','q']:
            print('Goodbye!')
            break

        stream_graph_updates(user_input)
    except:
        user_input = "What do you know about Fortaleza Esporte Clube?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
