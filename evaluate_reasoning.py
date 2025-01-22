import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import json

# # Ensure that the necessary NLTK data is downloaded
# nltk.download('punkt')
# nltk.download('punkt_tab')

def calculate_bleu_score(model_reasoning, ground_truth_reasoning):
    # Tokenize the model's reasoning and the ground truth reasoning
    model_tokens = nltk.word_tokenize(model_reasoning.lower())
    ground_truth_tokens = nltk.word_tokenize(ground_truth_reasoning.lower())
    
    # Calculate BLEU score (considering unigram to bigram matches)
    score = sentence_bleu([ground_truth_tokens], model_tokens, weights=(0.5, 0.5))
    return score

def calculate_rouge_score(model_reasoning, ground_truth_reasoning):
    # Create a scorer object for ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    scores = scorer.score(ground_truth_reasoning, model_reasoning)
    return scores

# Sample dataset
with open("results.json", "r") as f:
    data = json.load(f)

def evaluate_scores(data):
    for entry in data:
        if "evaluation_metrics" in entry:
            continue
        model_reasoning = entry["model_reasoning"]
        ground_truth_reasoning = entry["reasoning_ground_truth"]
        
        # Calculate BLEU Score
        bleu_score = calculate_bleu_score(model_reasoning, ground_truth_reasoning)
        
        # Calculate ROUGE Score
        rouge_scores = calculate_rouge_score(model_reasoning, ground_truth_reasoning)
        
        print(f"Question: {entry['question']}")
        print(f"BLEU Score: {bleu_score}")
        print(f"ROUGE Scores: {rouge_scores}")
        print('-' * 50)

# Run the evaluation for all entries in the data
evaluate_scores(data)
