from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
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
        print(result['answer'])
        return result

    def scratch(self):
        if len(self.chat_history) > 0:
            self.chat_history.pop()
            return True
        return False

    def reset(self):
        self.chat_history = []

    def history(self):
        return self.chat_history


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    db = FAISS.load_local("faiss_store", OpenAIEmbeddings())
    chat_bot = ConversationalQA(OpenAI(temperature=0, streaming=True), db.as_retriever())

    while True:
        print("\033[96muser:\033[00m ", end="")
        prompt = input()
        print("\033[93mbot:\033[00m ", end="")
        if prompt == "s":
            scratched = chat_bot.scratch()
            print("Latest question scratched" if scratched else "Nothing to scratch")
        elif prompt == "r":
            chat_bot.reset()
            print("Conversation reset")
        elif prompt == "q":
            print("Terminating")
            break
        elif prompt == "h":
            print(chat_bot.history())
        else:
            chat_bot.ask(prompt)