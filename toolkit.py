from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub

# 1. Carregar o documento PDF
pdf_path = "/home/claudio/Tutoriais/langchain_experimento/feriados.pdf"  # Substitua pelo caminho correto
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Dividir o texto em chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# 3. Criar embeddings e vetorstore
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(texts, embeddings)

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



