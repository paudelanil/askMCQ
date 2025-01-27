
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",))
sys.path.append(project_root)


from src.utils.chroma_db import ChromaDB, Splitter
from langchain_openai import OpenAIEmbeddings



from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Step 1: Initialize SentenceTransformer embeddings with LangChain
# model_name = "abhinand/MedEmbed-small-v0.1"
model_name = "abhinand/MedEmbed-base-v0.1"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
# embeddings = OpenAIEmbeddings(model='text-embedding-3-small')



# Step 2: Initialize Chroma vector store

chroma_db = ChromaDB(
    persist_directory='src/data/medembed_chroma_db',
    embeddings=embeddings
)

# Initialize Splitter
splitter = Splitter(chunk_size=500, chunk_overlap=50, min_words=10)

# Process PDFs
# pdf_splits = splitter.split_pdf_directory('data/raw/Collection')
# chroma_db.create_chroma_index(pdf_splits)

# Process Markdown files
markdown_splits = splitter.split_markdown_directory('/home/anil/Documents/CollegeWork8thSem/LLM/askMCQ/data/parsed_markdown')
chroma_db.upsert_index(markdown_splits)