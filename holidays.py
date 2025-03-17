from typing import Dict
from langchain_core import vectorstores
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

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
texts = text_splitter.split_documents(documents)

# 3. Criar embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='Jaume/gemma-2b-embeddings',
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings)

def parse_retriever_input(params: Dict):
    return params['messages'][-1].content

