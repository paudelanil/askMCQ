from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from src.vector_db import initialize_chroma
from src.embeddings import get_embedding_function
import yaml
from typing import List, Tuple

def load_config():
    """
    Loads configuration from config.yaml.
    """
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

class RetrievalAgent:
    def __init__(self):
        config = load_config()
        self.vector_store = initialize_chroma()  # Use LangChain's Chroma
        self.llm = ChatOpenAI(
            model=config["openai"]["llm_model"],
            api_key=config["openai"]["api_key"]
        )
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self):
        """
        Creates a prompt template for answering MCQs.
        """
        template = """
        You are an expert at answering multiple-choice questions (MCQs). Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}

        Options:
        {options}

        Provide the correct answer and a detailed reasoning for your choice.
        Correct Answer: 
        Reasoning:
        """
        return PromptTemplate(
            input_variables=["context", "question", "options"],
            template=template
        )

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieves relevant context for a query using LangChain's Chroma.
        """
        results = self.vector_store.similarity_search_with_score(query, k=top_k)
        return [(result[0].page_content, result[1]) for result in results]

    def answer_mcq(self, question: str, options: List[str]) -> str:
        """
        Answers an MCQ using retrieved context and reasoning.
        """
        # Retrieve relevant context
        context_results = self.retrieve_context(question)
        context = " ".join([chunk for chunk, _ in context_results])

        # Format options
        options_str = "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(options)])

        # Create a RunnableSequence for the LLM chain
        chain = (
            RunnablePassthrough.assign(context=lambda x: context)
            | self.prompt_template
            | self.llm
        )

        # Invoke the chain
        response = chain.invoke({
            "question": question,
            "options": options_str
        })

        # Add retrieved context to the response
        response += "\n\nRetrieved Context:\n"
        for chunk, score in context_results:
            response += f"- {chunk} (Score: {1 - score:.2f})\n"  # Convert distance to similarity score

        return response