# src/__init__.py

# Expose key functions/classes for easier imports
from src.utils.chroma_db import ChromaDB, Splitter, Retriever
from src.utils.query_processor import QueryProcessor

# Optional: Add package metadata
__version__ = "0.1"
__author__ = "Anil Paudel"