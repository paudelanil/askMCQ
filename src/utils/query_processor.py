from typing import Union, Optional
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Tuple
import re
from src.utils.chroma_db import ChromaDB, Retriever

class QueryProcessor:
    def __init__(self,retriever: Retriever,llm, chroma_db:ChromaDB):
        self.retriever = retriever
        self.llm = llm
        self.chroma_db = chroma_db

    def get_response(self,question: str, options: list, k: int = 5) -> str:
        """
        Generate a response to a multiple-choice question using the retriever, LLM, and LCEL chain.

        Args:
            retriever: The retriever object for fetching relevant context.
            llm: The language model for generating the response.
            question (str): The question to answer.
            options (list): A list of options for the question.
            k (int): The number of documents to retrieve.

        Returns:
            str: The generated response, including the correct answer and reasoning.
        """
        # Step 1: Retrieve relevant context
        # context, _ = self.retriever.similarity_search_withscore(question, k=k)

        context = self.retriever.rerank_documents(question, k=5)

        with open('prompt_1.txt','r') as f:
            prompt_text = f.read()
        # Step 2: Define the prompt template
        prompt_template = ChatPromptTemplate.from_template(prompt_text)
        # Step 3: Format the options
        formatted_options = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])

        # Step 4: Create the LCEL chain
        chain = (
            {
                "context": RunnableLambda(lambda x: context),  # Use the retrieved context
                "question": RunnablePassthrough(),  # Pass the question directly
                "options": RunnableLambda(lambda x: formatted_options),  # Format the options
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

        # Step 5: Invoke the chain and return the response
        response = chain.invoke(question)
        return response, context
    
    def process_questions(self,question_data:List[Dict]):

        # Process a list of questions and save the results
        results = []
        for i, question_data in enumerate(question_data):
            try:               

                response,context = self.get_response(question_data["question"], question_data["options"])

                # Add the retrieved documents to the question data
                model_letter, model_idx = self.parse_model_response(response)
                reasoning = response.split("Reasoning:")[1].strip() if "Reasoning:" in response else "No reasoning provided."

                results.append({
                    "question": question_data["question"],
                    "options": question_data["options"],
                    "correct_answer_text": question_data["correct_answer_text"],
                    "correct_answer_idx": question_data["correct_answer_idx"],
                    "model_answer": model_letter,
                    "model_answer_idx": model_idx,
                    "model_reasoning": reasoning,
                    "reasoning_ground_truth": question_data["reasoning"],
                    "context": context,
                    "ground_truth_context":question_data['ground_truth_context'],                
                })
                print(f"Processed question {i + 1}")
                
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                results.append(None)
        return results

    def parse_model_response(self, response_text: str) -> Tuple[Optional[str], Optional[int]]:
        """
        Extracts the model's letter answer and converts it to an index.
        """
        match = re.search(r'Correct Answer:\s*([A-D])', response_text, re.IGNORECASE)
        if not match:
            return None, None

        letter = match.group(1).upper()
        letter_to_idx = {'A': 0, 'B': 1,'C':2, 'D': 3}
        return letter, letter_to_idx.get(letter)