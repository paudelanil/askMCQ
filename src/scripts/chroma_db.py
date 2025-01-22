# scripts/chroma_db.py

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def create_chroma_index(splits, persist_directory):
    """
    Create a ChromaDB index from the processed splits.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectorstore.persist()
    return vectorstore

def init_chroma(persistent_directory):
    """
    Initialize the Chroma vector store.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("loading existing chroma DB Index")
    return Chroma( persist_directory=persistent_directory,
        embedding_function=embeddings)
    

def upsert_index(vectorstore, splits):
    """
    Update the ChromaDB index with new splits.
    """
    
    vectorstore.add_documents(documents =splits)
    vectorstore.persist()
    return vectorstore