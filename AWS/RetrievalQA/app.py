from flask import Flask, request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
import openai

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("faiss_store", embeddings)
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0, streaming=True),
    chain_type="stuff",
    retriever=db.as_retriever(),
    input_key="question",
)

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
    app.run(host='0.0.0.0', port=5000)