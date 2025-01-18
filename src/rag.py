from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from src.vector_db import initialize_chroma
import yaml

def load_config():
    """
    Loads configuration from config.yaml.
    """
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

class RAGSystem:
    def __init__(self):
        config = load_config()
        self.vector_store = initialize_chroma()
        self.llm = ChatOpenAI(
            model=config["openai"]["llm_model"],
            openai_api_key=config["openai"]["api_key"]
        )

    def answer_question(self, question: str):
        """
        Answers a question using the RAG system.
        """
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        return qa_chain.run(question)