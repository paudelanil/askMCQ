# utils/parser.py

import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

def parse_pdfs_to_markdown(input_dir, output_dir, parser_config):
    """
    Parse all PDFs in the input directory to Markdown and save them in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Configure the LlamaParse instance
    parser = LlamaParse(**parser_config)

    # Define a file extractor mapping file extensions to parsers
    file_extractor = {".pdf": parser}

    # Get a list of all PDF files in the input directory
    pdf_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pdf")]

    # Process each PDF file
    for pdf_file in pdf_files:
        try:
            print(f"Processing file: {pdf_file}")

            # Use SimpleDirectoryReader to parse the PDF file
            documents = SimpleDirectoryReader(
                input_files=[pdf_file],  # List of files to process
                file_extractor=file_extractor,
            ).load_data()

            # Extract the base name of the PDF file (without extension)
            base_name = os.path.splitext(os.path.basename(pdf_file))[0]

            # Define the output Markdown file path
            md_file_path = os.path.join(output_dir, f"{base_name}.md")

            # Write the parsed content to the Markdown file
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                for doc in documents:
                    # Write the text content of each document to the Markdown file
                    md_file.write(doc.text + "\n\n")  # Add extra newlines for separation

            print(f"Saved parsed content to: {md_file_path}")

        except Exception as e:
            print(f"Error processing file {pdf_file}: {e}")

    print("All files processed.")