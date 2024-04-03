from src.helper import *
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


document = load_repo("repo/")
text_chunks = text_splitter(document)
embeddings = load_embedding()

vectordb = Chroma.from_documents(documents=document, embedding=embeddings, persist_directory='./db')
vectordb.persist()
