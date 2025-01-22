import os
import json
from src.retrieval_agent import RetrievalAgent
from src.mutliquery_retreival import MultiQueryRetrievalAgent
from sklearn.metrics import confusion_matrix
import pandas as pd


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
    agent = RetrievalAgent(forEvaluate=True)
    results = agent.process_questions(question_data)
    
    
    with open(destinaion, "w") as f:
        json.dump(results, f, indent=2)
    # Evaluate the model
    evaluation_metrics = evaluate_model(results)

    # Add evaluation metrics to the results
    results.append({
        "evaluation_metrics": evaluation_metrics
    })

    with open('after_evaluation_similarity.json', "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to ", destinaion)

    return results


    
if __name__ == "__main__":
    evaluate_questions('question_with_gt_context_update.json','final_response_similartyscore.json')