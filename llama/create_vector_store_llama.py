from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
import os
import openai
import argparse

openai.api_key = os.getenv("OPENAI_API_KEY")

print("LlamaCpp embeddings") # TODO: LlamaCpp embeddings
embeddings = LlamaCppEmbeddings(model_path="/home/poludmik/virtual_env_project/llama/llama-2-7b-chat/ggml-model-q4.bin", n_ctx=2048)
# query_result = embeddings.embed_query("The fact that artificial intelligence has learned to draw is nothing. Think about what will happen when he is not accepted into the Vienna Academy of Arts.")
# print(len(query_result)) 
paths = ['../pdfs/Manual.pdf', ] # "../pdfs/Orwe1984-text20.pdf"

print("Loading documents")
pages = []
for path in paths:
    loader = UnstructuredPDFLoader(path) # Quite good, but slow
    #loader = PyPDFium2Loader(path)
    #loader = PDFMinerLoader(path)
    #loader = PyMuPDFLoader(path) # Fast
    #loader = PDFPlumberLoader(path)

    local_pages = loader.load_and_split()
    pages.extend(local_pages)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

# print("Start:")
# query_result = embeddings.embed_query(texts[50].page_content)
# print(len(query_result))

# Create vector database
print("Generating database")
db = FAISS.from_documents(texts, embeddings)
store_path = "faiss_store_llama_2-7b"
db.save_local(store_path)
print(f"Database stored in {store_path} folder")
