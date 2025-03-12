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


llamafile = Llamafile()

llamafile.invoke('Here is a recipe for pizza:')
