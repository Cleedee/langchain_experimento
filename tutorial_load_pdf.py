# Tutorial https://python.langchain.com/docs/tutorials/retrievers/
# Data da consulta 9/3/2025
# Nesse tutorial temos o uso de documentos e carregadores de documentos,
# separadores de texto, envelopadores (embeddings),
# armazéns de vetores (vector stores) e (recuperador) retrievers

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
import dotenv

dotenv.load_dotenv()

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def create_document_base(embeddings):
    file_path = "./data/nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    print(len(docs))
    print(f"{docs[0].page_content[:200]}\n")
    print(docs[0].metadata)

    # Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True 
    )
    all_splits = text_splitter.split_documents(docs)

    print("Total de Splits", len(all_splits))


    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Tamanho dos vetores gerados {len(vector_1)}\n")
    print(vector_1[:10])

    client = get_client_vectorstore()
    vector_store = get_vector_store(client, embeddings)
    ids = vector_store.add_documents(documents=all_splits)

def get_client_vectorstore():
    # 4. Configuração do vectorstore
    url = "http://localhost:6333"
    client = QdrantClient(url=url)
    if not client.collection_exists("tutorial-load-pdf"):
        client.create_collection(
            collection_name='tutorial-load-pdf',
            vectors_config=models.VectorParams(
                size=768, 
                distance=models.Distance.COSINE
            )
        )
    return client

def get_vector_store(client, embeddings):
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="tutorial-load-pdf",
        embedding=embeddings,
    )
    return vector_store


client = get_client_vectorstore()
vector_store = get_vector_store(client, embeddings)

#results = vector_store.similarity_search(
#    'How many distribution centers does Nike have in the US?'
#)

#print(results[0])

# Note that providers implement different scores; the score here
# is a distance metric that varies inversely with similarity.

#results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
#doc, score = results[0]
#print(f"Score: {score}\n")
#print(doc)

#embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")

#results = vector_store.similarity_search_by_vector(embedding)
#print(results[0])

retriever = vector_store.as_retriever(
    search_type = 'similarity',
    search_kwargs = { 'k': 1 },
)

retriever.batch([
    "How many distribution centers does Nike have in the US?",
    "When was Nike incorporated?",
])
