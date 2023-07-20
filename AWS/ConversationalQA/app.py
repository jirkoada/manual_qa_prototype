from flask import Flask, request
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
import os
import openai


class ConversationalQA:
    def __init__(self, embeddings, retriever):
        self.cqa = ConversationalRetrievalChain.from_llm(embeddings, retriever)
        self.chat_history = []

    def ask(self, question):
        result = self.cqa({"question": question, "chat_history": self.chat_history})
        self.chat_history.append((question, result['answer']))
        return result['answer']

    def scratch(self):
        if len(self.chat_history) > 0:
            self.chat_history.pop()
            return True
        return False

    def reset(self):
        self.chat_history = []

    def history(self):
        return self.chat_history
    

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
db = FAISS.load_local("faiss_store", OpenAIEmbeddings())
chat_bot = ConversationalQA(OpenAI(temperature=0, streaming=True), db.as_retriever())

# Route for handling the question answering request
@app.route('/answer', methods=['POST'])
def answer():
    # Retrieve the question from the AJAX request
    question = request.form['question']

    # Process the question and generate the answer using your LangChain model
    # Add your code here to handle the question and generate the answer

    if question == "s":
        scratched = chat_bot.scratch()
        ans = "Latest question scratched" if scratched else "Nothing to scratch"
    elif question == "r":
        chat_bot.reset()
        ans = "Conversation reset"
    elif question == "h":
        ans = str(chat_bot.history())
    else:
        ans = chat_bot.ask(question)

    # Return the answer as a response
    return {'answer': ans}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)