import getpass
import os
from typing import Annotated
from typing_extensions import TypedDict

import dotenv
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.tools import Tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.chat_models import  init_chat_model
import requests
from bs4 import BeautifulSoup

CONTACTS_PAGE_URL = "https://www.trt22.jus.br/informes/agenda-de-contatos"  

dotenv.load_dotenv()

def extract_phone_number(unit_name: str) -> str:
    try:
        # Faz a requisição HTTP para a página de contatos
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; LangchainBot/1.0)'}
        response = requests.get(CONTACTS_PAGE_URL, headers=headers, timeout=10)
        response.raise_for_status()  # Verifica se a requisição foi bem-sucedida

        # Parseia o conteúdo HTML da página
        soup = BeautifulSoup(response.text, 'html.parser')

        # Encontra todas as tabelas ou seções que contêm os contatos
        # (Ajuste o seletor conforme a estrutura da página)
        contact_tables = soup.find_all('table')  # Exemplo: busca todas as tabelas

        results = []
        for table in contact_tables:
            rows = table.find_all('tr')  # Encontra todas as linhas da tabela
            for row in rows:
                cells = row.find_all('td')  # Encontra todas as células da linha
                if len(cells) >= 2:  # Assume que a primeira coluna é o nome e a segunda é o telefone
                    name = cells[0].get_text(strip=True).lower()
                    phone = cells[1].get_text(strip=True)

                    # Verifica se o nome da unidade corresponde à consulta
                    if unit_name.lower() in name:
                        results.append(f"{name.capitalize()}: {phone}")

        if results:
            return "\n".join(results)
        return f"Nenhum contato encontrado para '{unit_name}'."

    except requests.exceptions.RequestException as e:
        return f"Erro na requisição: {str(e)}"
    except Exception as e:
        return f"Erro: {str(e)}"

phone_extractor_tool = Tool.from_function(
    func=extract_phone_number,
    name="PhoneExtractor",
    description="Busca números de telefone de uma unidade administrativa na página de contatos do TRT22. Recebe o nome da unidade como entrada."
)

modelo = init_chat_model("llama3-8b-8192", model_provider="groq")

wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

tools = [wikipedia_tool, phone_extractor_tool]

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
        {"messages": [("user", user_input)]}, config, stream_mode="values"
    )
    for event in events:
        event["messages"][-1].pretty_print()

