# src/__init__.py

# Expose key functions/classes for easier imports
from .pdf_parser import parse_pdf, split_text
from .embeddings import get_embedding_function
from .vector_db import initialize_chroma, upsert_documents, query_vector_store

# Optional: Add package metadata
__version__ = "0.1"
__author__ = "Anil Paudel"