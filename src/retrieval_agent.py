from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import json
from typing import List, Dict
import yaml
import re


class RetrievalAgent:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the RetrievalAgent with configuration and components.
        """
        self.config = self._load_config(config_path)
        self.vector_store = self._initialize_chroma()
        self.llm = ChatOpenAI(
            model=self.config["openai"]["llm_model"],
            api_key=self.config["openai"]["api_key"],
            temperature=0.1
        )
        self.chain = self._create_chain()

    def _load_config(self, config_path: str) -> Dict:
        """
        Loads configuration from a YAML file.
        """
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _initialize_chroma(self):
        """
        Initializes the Chroma vector store.
        """
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return Chroma(
            persist_directory=self.config["chroma"]["persist_directory"],
            embedding_function=embeddings
        )

    def _create_chain(self):
        """
        Creates the LCEL chain for question answering.
        """
        prompt_template = ChatPromptTemplate.from_template("""
        You are an expert at answering multiple-choice questions (MCQs). Use the following context to answer the question.

        Context:
        {context}

        Question:
        {question}

        Options:
        {options}

        Provide the correct answer as a single letter (A-D) and a detailed reasoning for your choice.
        Correct Answer: 
        Reasoning:
        """)

        return (
            {
                "context": RunnableLambda(lambda x: self.vector_store.similarity_search(x["question"], k=3)),
                "question": RunnablePassthrough(),
                "options": RunnableLambda(lambda x: "\n".join([f"{i+1}. {opt}" for i, opt in enumerate(x["options"])])),
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )
    def parse_model_response(self,response_text):
        """Extracts the model's letter answer and converts to index"""
        match = re.search(r'Correct Answer:\s*([A-D])', response_text, re.IGNORECASE)
        if not match:
            return None, None
        
        letter = match.group(1).upper()
        letter_to_idx = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
        return letter, letter_to_idx.get(letter)


    def process_questions(self, questions_data: List[Dict]):
        """
        Processes a list of questions and saves the results.
        """
        results = []
        for question_data in questions_data:
            try:
                # Retrieve context
                retrieved_docs = self.vector_store.similarity_search_with_score(question_data["question"], k=3)
                # Join document content and scores into a single string
                context = "\n".join([f"{doc.page_content} (Score: {score:.4f})" for doc, score in retrieved_docs])
                # Print retrieved context
                print(f"\nRetrieved Context for Question: {question_data['question']}")
                print(context)
                print("=" * 50)

                # Invoke the chain
                response = self.chain.invoke(question_data)
                print(response)

                # Parse the response
                # model_answer = response.split("Correct Answer:")[1].split()[0].strip()
                model_letter, model_idx = self.parse_model_response(response)
                reasoning = response.split("Reasoning:")[1].strip()

                results.append({
                "question": question_data["question"],
                "context": context,
                "model_answer": model_letter,
                "model_answer_idx": model_idx,
                "correct_answer_text": question_data["correct_answer_text"],
                "correct_answer_idx": question_data["correct_answer_idx"],
                "reasoning": reasoning,
                "is_correct": None  # Will be updated during evaluation
            })
            except Exception as e:
                print(f"Error processing question: {question_data['question']}")
                print(f"Error: {str(e)}")

        return results
    
