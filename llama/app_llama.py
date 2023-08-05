from flask import Flask, render_template, request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.llms import LlamaCpp
import os
import argparse
import openai


app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")

print("Using OpenAI embeddings")
embeddings = OpenAIEmbeddings()
    
db = FAISS.load_local("faiss_store_llama", embeddings)

print("Using Llama2 for QA")
llm = LlamaCpp(model_path="/home/poludmik/virtual_env_project/llama/llama-2-7b-chat/ggml-model-q4.bin", n_ctx=2048)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(),
    input_key="question",
)

print("Prompt template:\n", qa.combine_documents_chain.llm_chain.prompt.template)

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the question answering request
@app.route('/answer', methods=['POST'])
def answer():
    # Retrieve the question from the AJAX request
    question = request.form['question']

    # Process the question and generate the answer using your LangChain model
    # Add your code here to handle the question and generate the answer
    ans = qa.run(question)

    # Return the answer as a response
    return {'answer': ans}

if __name__ == '__main__':
    print("Starting test client.")
    app.run()