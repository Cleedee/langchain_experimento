import os
from typing import Annotated
from typing_extensions import TypedDict

import dotenv
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.tools import Tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from toolkit.portal import phone_extractor_tool, office_hours_extractor_tool

dotenv.load_dotenv()



wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

tools = [wikipedia_tool, phone_extractor_tool, office_hours_extractor_tool]

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)


groq_api = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(api_key=groq_api, model="Gemma2-9b-It")
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

memory = MemorySaver()
graph_memory = graph_builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Goodbye!")
        break

    events = graph_memory.stream(
        {
            # "messages": [("user", user_input)]
            "messages": [{'role': 'user', 'content': user_input}]
        },
        config, 
        stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()

