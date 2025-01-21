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