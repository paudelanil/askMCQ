
import os
import sys
from parser import parse_pdfs_to_markdown
from splitter import split_markdown_files
from chroma_db import create_chroma_index

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

# Directories
INPUT_DIR = "data/raw/NewMedCollection"  # Directory containing PDF files
OUTPUT_DIR = "data/parsed_markdown"  # Directory to save parsed Markdown files
CHROMA_PERSIST_DIR = "data/chroma_db"  # Directory to persist ChromaDB

# LlamaParse Configuration
PARSER_CONFIG = {
    "result_type": "markdown",  # Output format
    "num_workers": 8,  # Number of workers for parallel processing
    "verbose": True,  # Enable verbose logging
    "language": "en",  # Language of the document
    "show_progress": True,  # Show progress bar
}

# Text Splitting Configuration
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),  # Header level 1
    ("##", "Header 2"),  # Header level 2
    ("###", "Header 3"),  # Header level 3
]

CHUNK_SIZE = 300  # Size of each split chunk
CHUNK_OVERLAP = 30  # Number of overlapping characters between chunks
MIN_WORDS_PER_CHUNK = 10  # Minimum words per chunk



def main():
    # Step 1: Parse PDFs to Markdown
    # parse_pdfs_to_markdown(INPUT_DIR, OUTPUT_DIR, PARSER_CONFIG)

    # Step 2: Split Markdown files into chunks
    splits = split_markdown_files(
        OUTPUT_DIR, HEADERS_TO_SPLIT_ON, CHUNK_SIZE, CHUNK_OVERLAP, MIN_WORDS_PER_CHUNK
    )

    # Step 3: Create ChromaDB index
    vectorstore = create_chroma_index(splits, CHROMA_PERSIST_DIR)
    print(f"ChromaDB index created with {len(splits)} chunks.")

if __name__ == "__main__":
    main()