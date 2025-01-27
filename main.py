
import json
from src.utils.chroma_db import ChromaDB, Splitter, Retriever
from src.utils.query_processor import QueryProcessor
from src.utils.evaluation import evaluate_model
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings


import json
import os
import pandas as pd

load_dotenv(override=True)

# model_name = "abhinand/MedEmbed-small-v0.1"
model_name = "abhinand/MedEmbed-base-v0.1"
embeddings = HuggingFaceEmbeddings(model_name=model_name)


def main():
    chataopenai = ChatOpenAI(
            model = 'gpt-4o',
            temperature=0.1)

    # embeddings = OpenAIEmbeddings(
    #     model = 'text-embedding-3-small'
    # )

   

    # chroma_db = ChromaDB(persist_directory='data/chroma_db', 
    #                     embeddings=embeddings)

    chroma_db = ChromaDB(
    persist_directory='src/data/medembed_chroma_db',
    embeddings=embeddings
)


    retriever = Retriever(chroma_db.vectorstore)
    queryprocessor = QueryProcessor(retriever=retriever,llm=chataopenai,chroma_db= chroma_db)

    # response = queryprocessor.get_response(question,option)


    with open('question_collection.json', 'r') as file:
        questions = json.load(file)

    # print(questions)
    response = queryprocessor.process_questions(questions)

    with open('outputs/gpt4o_openaiembed_similarity.json', "w") as f:
        json.dump(response, f, indent=2)
        
if __name__ == "__main__":
    # evaluate_questions('question_with_gt_context_update.json','ouptput_3_rerank.json')
    main()
    