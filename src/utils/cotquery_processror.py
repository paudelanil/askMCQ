from src.utils.query_processor import QueryProcessor
from collections import Counter
from typing import List, Tuple, Dict

from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class CoTQueryProcessor(QueryProcessor):
    def __init__(self, retriever, llm, chroma_db, num_samples: int = 5, temperature: float = 0.7):
        super().__init__(retriever, llm, chroma_db)
        self.num_samples = num_samples
        self.temperature = temperature

    def get_response(self, question: str, options: List[str], k: int = 5) -> Tuple[str, List[str]]:
        # Retrieve context
        question_with_options = f"{question}\n" + "\n".join(options)  # this results in a better context retreiving rathen than just sending queries

        context,__ = self.retriever.similarity_search_withscore(question_with_options, k=k)        
        
        # Load improved prompt
        with open('src/utils/prompts/prompts_cot_deepseek.txt', 'r') as f:
            prompt_text = f.read()
            
        formatted_options = "\n".join([f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)])
        answers = []
        responses = []
        # Generate multiple reasoning paths
        for i in range(self.num_samples):
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
            # Generate with temperature for diversity
            response = chain.invoke(question)
            responses.append(response)

            # print(f"response{i}:{'--'*50} \n {response}")
            answer = self.parse_model_response(response)
            if answer:
                answers.append(answer)
        
        # Majority voting
        most_common = self.get_most_consistent_answer(answers)


        return most_common, context, responses

    def process_questions(self, question_data: List[Dict]) -> List[Dict]:

        results = []
        for i, question_dict in enumerate(question_data):
            
            try:
                question = question_dict["question"]
                options = question_dict["options"]
                answer, context,responses = self.get_response(question, options)
                model_letter, model_idx = answer[0],answer[1]

                results.append({
                        "question": question_dict["question"],
                        "options": question_dict["options"],
                        "correct_answer_text": question_dict["correct_answer_text"],
                        "correct_answer_idx": question_dict["correct_answer_idx"],
                        "model_answer": model_letter,
                        "model_answer_idx": model_idx,
                        "context": context,
                        "ground_truth_context":question_dict['ground_truth_context'], 
                        "model_response":responses        
                    })
            
                print(f"Processed question {i + 1}")
                
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                results.append(None)
        return results
       



    def get_most_consistent_answer(self, answers: List[str]) -> str:
        return Counter(answers).most_common(1)[0][0] if answers else "Unknown"