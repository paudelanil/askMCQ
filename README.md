# askMCQ

This project focuses on creating and managing vector embeddings for medical document processing and semantic search. It supports querying via question-based inputs and leverages embeddings to deliver highly relevant responses. The repository is designed with a structured pipeline for ingesting data, creating embeddings, and processing queries.

## Key Components

### Scripts

- **ingest.py**: Used to create vector embeddings from processed data. This is the starting point for embedding generation and populating the vector database. After selecting your choice of embeddings and persist directory for Chroma vector db, you will get the indexed database.
- **parser.py:** Used to parse pdf to markdown format levaring llamaparser. Use your `llama cloud api key` in .env

### Utils

- **chroma\_db.py**: Contains helper classes and methods for interacting with the vector database and retriever tecnhiques.
- **query\_processor.py** : Contains classes making query processing easier. Includes question parser, llm invoke chain, post processing logics.
- **evaluation.py**: Includes evaluation logic to assess the performance of the embeddings and retrieval mechanisms.

### Main Logic

- **main.py**: Core script to run the system. It ties together embedding creation, query processing, and response generation.

###

## Workflow

1. **Prepare Data**:

   - Place raw documents in the `data/raw` folder.
   - Use `src/scripts/parser.py` to process documents into markdown format and store them in `data/parsed_markdown`.

2. **Create Embeddings**:

   - Run `ingest.py` to create vector embeddings using the parsed data. These embeddings are stored in the `chroma_db` directory.

3. **Query Processing**:

   - Use `main.py` to:
     - Process questions using the `process_question` method.
     - Generate responses via `generate_response` with question and options inputs.

4. **Evaluate Results**:

   - Use the `Evaluation.ipynb` notebook to evaluate the response as per accuracy, confusion matrix, similarity of reasoning and context retrieved with ground truth.

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Example Usage

### Generating Embeddings

Run the following command to generate embeddings:

```bash
python src/scripts/ingest.py
```

### Querying

Use `main.py` to process questions:

```bash
python main.py
```

## Outputs

Results and intermediate outputs are stored in the `outputs` directory. You can analyze these to assess model performance or further refine processing logic.

## Documentation

For additional details, refer to `docs/askMCQ_Presentation.pdf`.

