import os
from dotenv import load_dotenv
from langchain_community.document_loader import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

load_dotenv()


