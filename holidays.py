import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain_community.tools import Tool
import dotenv
import streamlit as st

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
texts = text_splitter.split_documents(documents)

# 3. Criar embeddings
embedding_model_name = "Jaume/gemma-2b-embeddings"
embedding_model_kwargs = {'device': 'cpu'}
embedding_encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=embedding_encode_kwargs
)
#
#model_name = "BAAI/bge-large-en"
model_name = "google/gemma-2-9b-it"
#model_name = "google/flan-t5-large"
model_kwargs = {'device': 'cpu', 'temperature': 0.7 }
encode_kwargs = {'normalize_embeddings': False}

# 4. Configuração do vectorstore
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db",
#    force_recreate = True
)

# 5. Configurar o modelo de linguagem

model = HuggingFaceHub(
    repo_id=model_name,
    model_kwargs=model_kwargs
)

#retriever = qdrant.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=model, 
    chain_type='stuff', 
    retriever=qdrant.as_retriever()
)

def questoes_feriados(question: str):
    resposta = qa_chain.invoke(question)
    print(f"Questão: {question}")
    print(f"Tipo da Resposta: {type(resposta)}")
    return resposta

holidays_tool = Tool.from_function(
    func=questoes_feriados,
    name="HolidaysExtrator",
    description=(
        "Busca feriados que afetam os expedientes do tribunal regional "
        "do trabalho da 22ª região. Recebe uma pergunta como entrada."
    )
)
