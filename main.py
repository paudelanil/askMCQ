
import json
from src.utils.chroma_db import ChromaDB, Splitter, Retriever
from src.utils.query_processor import QueryProcessor
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings

from sklearn.metrics import confusion_matrix

import json
import os
import pandas as pd

load_dotenv(override=True)

model_name = "abhinand/MedEmbed-small-v0.1"
embeddings = HuggingFaceEmbeddings(model_name=model_name)


def evaluate_model(questions_data):
    """
    Evaluates the model's performance and computes accuracy and confusion matrix.
    """
    total = 0
    correct = 0
    y_true = []  # Ground truth labels
    y_pred = []  # Predicted labels

    for q in questions_data:
        # Ensure required fields exist
        if "correct_answer_idx" not in q or "model_answer_idx" not in q:
            print(f"Skipping question (missing fields): {q['question']}")
            continue

        # Compare with ground truth
        is_correct = (q["model_answer_idx"] == q["correct_answer_idx"])
        q["is_correct"] = is_correct

        # Update metrics
        total += 1
        if is_correct:
            correct += 1

        # Append to confusion matrix data
        y_true.append(q["correct_answer_idx"])
        y_pred.append(q["model_answer_idx"])

    # Compute confusion matrix
    confusion = confusion_matrix(y_true, y_pred, labels=[0, 1,2, 3])
    confusion_df = pd.DataFrame(
        confusion,
        index=["True A", "True B", "True C", "True D"],
        columns=["Pred A", "Pred B", "Pred C", "Pred D"]
    )

    return {
        "accuracy": correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "confusion_matrix": confusion_df.to_dict()
    }

def evaluate_questions(source,destinaion):
    """
    Evaluates a single question and returns the results.
    """

    with open(source, "r") as f:
        question_data = json.load(f)

    # agent = MultiQueryRetrievalAgent(forEvaluate=True)
    agent = RerankRetrieval(forEvaluate=True)
    results = agent.process_questions(question_data)
    
    
    with open(destinaion, "w") as f:
        json.dump(results, f, indent=2)
    # Evaluate the model
    evaluation_metrics = evaluate_model(results)

    # Add evaluation metrics to the results
    results.append({
        "evaluation_metrics": evaluation_metrics
    })

    with open(f'{destinaion}_after_evaluation_similarity.json', "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to ", destinaion)

    return results

def main():
    chataopenai = ChatOpenAI(
            model = 'gpt-4o-mini',
            temperature=0.1)

    # embedding = OpenAIEmbeddings(
    #     model = 'text-embedding-3-small'
    # )

   

    # chroma_db = ChromaDB(model='text-embedding-3-small',
    #                     persist_directory='data/chroma_db', 
    #                     embeddings=embedding)

    chroma_db = ChromaDB(
    persist_directory='src/data/medembed_chroma_db',
    embeddings=embeddings
)


    retriever = Retriever(chroma_db.vectorstore)
    queryprocessor = QueryProcessor(retriever=retriever,llm=chataopenai,chroma_db= chroma_db)

    # question = "Why we should split?"
    # option  = ["Apple","Banana","Cherry","Date"]
    # response = queryprocessor.get_response(question,option)


    with open('question_collection.json', 'r') as file:
        questions = json.load(file)

    # print(questions)
    response = queryprocessor.process_questions(questions)


    # print(retriever.retrieve("What is the capital of France?"))
    # context, score = retriever.similarity_search_withscore("Spliter working?", k=2)
    # print(response)
    # eval_metric = evaluate_model(response)
    # response.append({"evaluation_metrics":eval_metric})  

    with open('outputs/v2/medembedd_similarity_retreiver_GPT4o.json', "w") as f:
        json.dump(response, f, indent=2)
        
if __name__ == "__main__":
    # evaluate_questions('question_with_gt_context_update.json','ouptput_3_rerank.json')
    main()
    