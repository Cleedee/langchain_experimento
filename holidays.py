from typing import Dict

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import MessagesState, StateGraph

import dotenv

dotenv.load_dotenv()

# 1. Carregar o documento PDF
# HOLIDAYS_FILE = r"C:\Users\claudio.torcato\Tutoriais\langchain_experimento\feriados.pdf"
HOLIDAYS_FILE = r"C:\Users\User\Projetos\tutoriais\langchain_experimento\data\feriados.pdf"
print("Caminho do PDF:", HOLIDAYS_FILE)
loader = PyPDFLoader(HOLIDAYS_FILE)
documents = loader.load()

# 2. Dividir o texto em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
all_splits = text_splitter.split_documents(documents)

# 3. Criar embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='Jaume/gemma-2b-embeddings',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

vectorstore = InMemoryVectorStore(embeddings)

_ = vectorstore.add_documents(documents=all_splits)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}
