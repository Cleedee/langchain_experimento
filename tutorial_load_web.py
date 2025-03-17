import os

from langchain_community.llms.llamafile import Llamafile
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.memory import ConversationBufferMemory

# https://medium.com/@Shamimw/langchain-document-loader-connecting-to-different-systems-76f197105dc6
# https://python.langchain.com/v0.1/docs/use_cases/question_answering/local_retrieval_qa/
# https://wiki.qt.io/Jom
# https://github.com/Mozilla-Ocho/llamafile

llamafile = Llamafile()

loader = WebBaseLoader("https://telegra.ph/Campanha-por-Escrito-de-Forbidden-Lands---Parte-4-03-11")

doc = loader.load()[0]

query = "Qual o nome meio-elfo?"

template = (
    "You are a helpful assistant that reads documents {documents}"
    " and answers the prompt: {query}."
)

prompt = ChatPromptTemplate.from_template(template=template)

chain = prompt | llamafile | StrOutputParser()

# Start a conversation by invoking the chain with the correct input variables
response = chain.invoke({
    "documents": doc,
    "query": query
})

print("--------------------------")
print()

print(response)

print()

print("--------------------------")
