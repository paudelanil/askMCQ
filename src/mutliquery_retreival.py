from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
import yaml
import re
import json
from typing import List, Dict, Optional, Tuple


class MultiQueryRetrievalAgent:
    def __init__(self, config_path: str = "config/config.yaml",forEvaluate=False):
        """
        Initializes the MultiQueryRetrievalAgent with configuration and components.
        """
        self.config = self._load_config(config_path)
        self.vector_store = self._initialize_chroma()
        self.llm = ChatOpenAI(
            model=self.config["openai"]["llm_model"],
            api_key=self.config["openai"]["api_key"],
            temperature=0.1
        )
        self.retriever = self._initialize_multiquery_retriever()
        self.chain = self._create_chain()
        self.forEvaluate = forEvaluate

    def _load_config(self, config_path: str) -> Dict:
        """
        Loads configuration from a YAML file.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            if not config.get("openai") or not config.get("chroma"):
                raise ValueError("Configuration file is missing required keys.")
            return config

    def _initialize_chroma(self) -> Chroma:
        """
        Initializes the Chroma vector store.
        """
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        return Chroma(
            persist_directory=self.config["chroma"]["persist_directory"],
            embedding_function=embeddings
        )

    def _initialize_multiquery_retriever(self):
        """
        Initializes the MultiQueryRetriever with the vector store.
        """
        return MultiQueryRetriever.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(),
            
        )
    def _create_chain_temp(self):
        with open('prompt_1.txt', 'r') as file:
            prompt_template = file.read()

        return (
            {
                "context": RunnableLambda(lambda x: self._retrieve_multiquery_context(x["question"])),
                "question": RunnablePassthrough(),
                "options": RunnableLambda(lambda x: "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(x["options"])])),
                "correct_answer_text": RunnablePassthrough(),
                "correct_answer_idx": RunnablePassthrough(),
                "reasoning": RunnablePassthrough(),
            }
            | ChatPromptTemplate.from_template(prompt_template)
            | self.llm
            |StrOutputParser()
    #         | RunnableMap({
    # "selected_answer": lambda x: json.loads(x.content).get("selected_answer"),
    # "reasoning": lambda x: json.loads(x.content).get("reasoning"),
    # "is_correct": lambda x: json.loads(x.content).get("selected_answer") == x["correct_answer_text"],
    # "evaluation": lambda x: (
    #     f"Ground Truth Reasoning: {x['reasoning']} \n"
    #     f"Model Reasoning: {json.loads(x.content).get('reasoning')} \n"
    #     f"Comparison: {'Matched' if json.loads(x.content).get('reasoning') == x['reasoning'] else 'Not Matched'}"
    # )
# })

        )
    def _create_chain(self):
        """
        Creates the chain for question answering.
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
                "context": RunnableLambda(lambda x: x["retrieved_docs"]),
                "question": RunnablePassthrough(),
                "options": RunnableLambda(lambda x: "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(x["options"])])),
            }
            | prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _retrieve_multiquery_context(self, query: str) -> str:
        """
        Retrieves context using the MultiQueryRetriever.
        """
        retreived_docs = self.retriever.invoke(query)
        print("length of retreived_docs",len(retreived_docs))

        if not retreived_docs:
            return "No relevant context found."
        return "\n".join([f"{doc.page_content}" for doc in retreived_docs])

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

    def process_questions(self, questions_data: List[Dict]) -> List[Dict]:
        """
        Processes a list of questions and saves the results.
        """
        results = []
        for i, question_data in enumerate(questions_data):
            try:
                # Retrieve context using multi-query retrieval
                context = self._retrieve_multiquery_context(question_data["question"])
                question_data['retireved_docs'] = context
                # Print retrieved context
                # print(f"\nRetrieved Context for Question: {question_data['question']}")
                # print(context)
                # print("=" * 50)

                # Invoke the chain
                response = self.chain.invoke(question_data)
               

                # Parse the response
                model_letter, model_idx = self.parse_model_response(response)
                reasoning = response.split("Reasoning:")[1].strip() if "Reasoning:" in response else "No reasoning provided."
                
                if self.forEvaluate:
                        
                    results.append({
                        "question": question_data["question"],
                        "context": context,
                        "ground_truth_context":question_data['ground_truth_context'],
                        "model_answer": model_letter,
                        "model_answer_idx": model_idx,
                        "correct_answer_text": question_data["correct_answer_text"],
                        "correct_answer_idx": question_data["correct_answer_idx"],
                        "model_reasoning": reasoning,
                        "reasoning_ground_truth": question_data["reasoning"],
                        "is_correct": None  # Will be updated during evaluation,
                    })
                else:
                    results.append({
                        "question": question_data["question"],
                        "context": context,
                        "model_answer": model_letter,
                        "model_answer_idx": model_idx,
                        "reasoning": reasoning
                    })

            except Exception as e:
                print(f"Error processing question: {question_data['question']}")
                print(f"Error: {str(e)}")
            print(f"Processed question {i+1}\n")
        return results