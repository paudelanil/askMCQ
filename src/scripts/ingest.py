
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",))
sys.path.append(project_root)


from src.utils.chroma_db import ChromaDB, Splitter
from langchain_openai import OpenAIEmbeddings


# Initialize embedding model
embedding = OpenAIEmbeddings(model='text-embedding-3-small')

# Initialize ChromaDB
chroma_db = ChromaDB(
    model='text-embedding-3-small',
    persist_directory='src/data/chroma_db',
    embeddings=embedding
)

# Initialize Splitter
splitter = Splitter(chunk_size=500, chunk_overlap=50, min_words=10)

# Process PDFs
# pdf_splits = splitter.split_pdf_directory('data/raw/Collection')
# chroma_db.create_chroma_index(pdf_splits)

# Process Markdown files
markdown_splits = splitter.split_markdown_directory('/home/anil/Documents/CollegeWork8thSem/LLM/askMCQ/data/parsed_markdown')
chroma_db.upsert_index(markdown_splits)