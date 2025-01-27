
from sklearn.metrics import confusion_matrix


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
