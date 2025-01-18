from langchain_chroma import Chroma
from src.embeddings import get_embedding_function

import yaml
from typing import List, Tuple

def load_config():
    """
    Loads configuration from config.yaml.
    """
    with open("config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

def initialize_chroma():
    """
    Initializes Chroma and creates/loads a collection.
    """
    config = load_config()
    embedding_function = get_embedding_function()

    vector_store = Chroma(
        collection_name = "mdeical-guidelines",
        persist_directory=config["chroma"]["persist_directory"],
        embedding_function=embedding_function)
    
    return vector_store   



def upsert_documents(vector_store, chunks: List[str]):
    """
    Upserts documents into Chroma.
    """
    vector_store.add_texts(chunks)
    

def query_vector_store(vector_store, query: str, top_k: int = 3):
    """
    Queries the Chroma vector store for similar documents.
    """
    results = vector_store.similarity_search(query, k=top_k)
    return results