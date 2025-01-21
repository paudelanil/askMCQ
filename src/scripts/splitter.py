
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
import os

def split_markdown_files(markdown_dir, headers_to_split_on, chunk_size, chunk_overlap, min_words):
    """
    Split Markdown files into chunks based on headers and recursive character splitting.
    """
    all_splits = []

    # Initialize splitters
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    # Process each Markdown file
    for md_file in os.listdir(markdown_dir):
        if md_file.endswith(".md"):
            md_file_path = os.path.join(markdown_dir, md_file)
            with open(md_file_path, "r", encoding="utf-8") as f:
                markdown_document = f.read()

            # Step 1: Split by headers
            header_splits = markdown_splitter.split_text(markdown_document)

            # Step 2: Filter small chunks
            filtered_splits = [
                split for split in header_splits
                if len(split.page_content.split()) >= min_words
            ]

            # Step 3: Further split by character
            final_splits = text_splitter.split_documents(filtered_splits)

            # Add source file metadata
            for split in final_splits:
                split.metadata["source"] = md_file

            all_splits.extend(final_splits)

    return all_splits