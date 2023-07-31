from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
import os
import openai
from ft_embs import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fasttext", default=False, action=argparse.BooleanOptionalAction, help="Use fasttext embeddings")
args = parser.parse_args([] if "__file__" not in globals() else None)

openai.api_key = os.getenv("OPENAI_API_KEY")

if args.fasttext:
    print("Using fastText embeddings")
    embeddings = FTembeddings()
    paths = ["pdfs/Orwe1984-text20.pdf"]
else:
    print("Using OpenAI embeddings")
    embeddings = OpenAIEmbeddings()
    paths = ['pdfs/Manual.pdf']

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

# Create vector database
print("Generating database")
db = FAISS.from_documents(texts, embeddings)
store_path = "faiss_store"
db.save_local(store_path)

print(f"Database stored in {store_path} folder")