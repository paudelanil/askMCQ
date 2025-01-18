import os
import json
from src.pdf_parser import parse_pdf, split_text
from src.vector_db import initialize_chroma, upsert_documents
from src.retrieval_agent import RetrievalAgent

def process_pdf(file_path):
    """
    Processes a PDF file and indexes it into Chroma.
    """
    # Parse PDF
    text = parse_pdf(file_path)
    
    # Split text into chunks
    chunks = split_text(text)
    
    # Initialize Chroma and upsert documents
    vector_store = initialize_chroma()
    upsert_documents(vector_store, chunks)

def load_questions(json_file_path: str):
    """
    Loads questions and options from a JSON file.
    """
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data

def main():
    # Directory containing PDFs
    # pdf_directory = "data/raw"

    # # Process each PDF (if not already processed)
    # for filename in os.listdir(pdf_directory):
    #     if filename.endswith(".pdf"):
    #         file_path = os.path.join(pdf_directory, filename)
    #         print(f"Processing: {filename}")
    #         process_pdf(file_path)

    # Initialize Retrieval Agent
    agent = RetrievalAgent()

    # Load questions from JSON file
    json_file_path = "question.json"  # Path to the JSON file
    questions_data = load_questions(json_file_path)

    # Answer each question
    for question_data in questions_data:
        question = question_data.get("question")
        options = question_data.get("options")

        if not question or not options:
            print(f"Skipping invalid question: {question_data}")
            continue

        print(f"\nQuestion: {question}")
        print("Options:")
        for i, option in enumerate(options, start=1):
            print(f"{i}. {option}")

        # Get answer with reasoning
        answer = agent.answer_mcq(question, options)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()