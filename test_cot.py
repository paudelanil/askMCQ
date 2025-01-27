
import json
from src.utils.chroma_db import ChromaDB, Splitter, Retriever
from src.utils.cotquery_processror import CoTQueryProcessor
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

chataopenai = ChatOpenAI(
            model = 'gpt-4o-mini',
            temperature=0.7)

chroma_db = ChromaDB(
persist_directory='src/data/medembed_chroma_db',
embeddings=embeddings
)

retriever = Retriever(chroma_db.vectorstore)

cot_processor = CoTQueryProcessor(retriever=retriever,llm=chataopenai,chroma_db= chroma_db,num_samples=3)
# question = "What is the most common cause of death in the United States?"

# options= ["Heart disease","Cancer","Stroke","Diabetes","Alzheimer's disease"]

# question="A 40-year-old male presents to the emergency department with a witnessed seizure lasting 7 minutes. He has no known history of epilepsy. According to the guidelines, what is the recommended initial therapy?",
# options=["Intravenous phenytoin", "Intramuscular midazolam", "Intravenous lorazepam", "Intravenous phenobarbital" ]


# print(cot_processor.get_response(question,options))



with open('question_collection.json', 'r') as file:
        questions = json.load(file)

    # print(questions)
response = cot_processor.process_questions(questions)


with open('cot_try_improved_context.json', "w") as f:
    json.dump(response, f, indent=2)
        

