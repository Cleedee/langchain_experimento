import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import dotenv

dotenv.load_dotenv()

# 1. Carregar o documento PDF
HOLIDAYS_FILE=r"C:\Users\User\Projetos\tutoriais\langchain_experimento\feriados.pdf"

print("Caminho do PDF:", HOLIDAYS_FILE)
loader = PyPDFLoader(HOLIDAYS_FILE)
documents = loader.load()

# 2. Dividir o texto em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

def create_vectorstore(texts):
    # 3. Criar embeddings e vetorstore
    #embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    embeddings = HuggingFaceEmbeddings(model_name="whereIsAI/UAE-Large-V1")
    vectorstore = FAISS.from_documents(texts, embeddings)
    #vectorstore = FAISS.from_texts(text=chunks, embedding=embeddings)

    vectorstore.save_local('faiss_default')

    return vectorstore

vectorstore = create_vectorstore(texts)

def create_conversation_chain(vectorstore=None):
    if not vectorstore:
        embeddings = HuggingFaceEmbeddings(model_name="whereIsAI/UAE-Large-V1")
        vectorstore = FAISS.load_local('faiss_default', embeddings=embeddings)
    llm = HuggingFaceHub(repo_id='google/flan-t5-large', model_kwargs={})
    memory = ConversationBufferMemory(memory_ref='chat_history')
    conversation_chain = ConversationRetrievalChain.from_text(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# 4. Configurar o modelo de linguagem (escolha uma das opções abaixo)

# Opção 1: Usando HuggingFace (gratuito)
model_name = "google/flan-t5-xxl"
model = HuggingFaceHub(
    repo_id=model_name,
    model_kwargs={"temperature":0.1, "max_length":512}
)

# 5. Criar a cadeia de Q&A
qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# 6. Fazer uma pergunta
pergunta = "Quais são os feriados de outubro?"
resultado = qa_chain({"query": pergunta})

print("Resposta:", resultado["result"])
print("\nFontes utilizadas:")
for doc in resultado["source_documents"]:
    print(f"- Página {doc.metadata['page']}: {doc.page_content[:100]}...")

