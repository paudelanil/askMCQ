from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter)
from PyPDF2 import PdfReader
from langchain.retrievers.multi_query import MultiQueryRetriever
from sentence_transformers import CrossEncoder
from typing import List, Dict, Tuple



import os

class ChromaDB:
    def __init__(self, model, persist_directory,embeddings):
        self.model = model
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.embeddings = embeddings
        self._init_chroma()  
            
    def _init_chroma(self):
        """
        Initialize the Chroma vector store.
        """
        if os.path.exists(self.persist_directory):
            print("loading existing chroma DB Index")
        


        self.vectorstore = Chroma( persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
           )


    def create_chroma_index(self, splits):
        """
        Create a ChromaDB index from the processed splits.
        """
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print("Chroma DB Index created")
        return self.vectorstore

    def upsert_index(self, splits):
        """
        Update the ChromaDB index with new splits.
        """
        self.vectorstore.add_documents(documents =splits)
        print("Chroma DB Index updated")
        return self.vectorstore



class Splitter:
    def __init__(self, chunk_size, chunk_overlap, min_words):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_words = min_words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3")
    ])

    def split_pdf(self, pdf_path):
        try:
            reader = PdfReader(pdf_path)

            metadatas = []
            texts = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:  # Check if text is not None or empty
                    texts.append(text)
                    metadatas.append({
                        "source_file": pdf_path,
                        "page_number": i + 1
                    })

            # Split text into chunks
            chunks = []
            for text, metadata in zip(texts, metadatas):
                # Split each page's text into chunks
                page_chunks = self.text_splitter.split_text(text)
                for chunk in page_chunks:
                    # Create a document for each chunk with the corresponding metadata
                    chunks.append({
                        "text": chunk,
                        "metadata": metadata
                    })

            # Filter chunks based on the minimum word count
            filtered_chunks = [chunk for chunk in chunks if len(chunk["text"].split()) >= self.min_words]

            return filtered_chunks
        
        except Exception as e:
            print(f"Error splitting PDF: {e}")
            return []
        


    def split_pdf_directory(self, directory):
        """
        Split all PDF files in the specified directory.
        """
        try:
            all_splits = []
            for file in os.listdir(directory):
                if file.lower().endswith(".pdf"):
                    print(f"Splitting PDF: {file}")
                    pdf_path = os.path.join(directory, file)
                    pdf_splits = self.split_pdf(pdf_path)
                    all_splits.extend(pdf_splits)
            return all_splits
        except Exception as e:
            print(f"Error splitting directory: {e}")
            return []
        
    
    
    def split_markdown(self,markdown_path):
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_document = f.read()

        # split by header first

        md_chunks = self.md_splitter.split_text(markdown_document)


        filtered_splits = [
            split for split in md_chunks
            if len(split.page_content.split()) >= self.min_words
        ]

        final_splits = self.text_splitter.split_documents(filtered_splits)

        # Add source file metadata
        for split in final_splits:
            split.metadata["source"] = markdown_path

        
        
        return final_splits
 

    def split_markdown_directory(self, directory):
        """
        Split all markdown files in the specified directory.
        """

    
        try:
            all_splits = []
            # Iterate over all files in the directory
            for file in os.listdir(directory):
                # Check if the file is a markdown file (case-insensitive)
                if file.lower().endswith(".md") or file.lower().endswith(".markdown"):
                    print(f"Splitting markdown file: {file}")  # Log the file being processed
                    markdown_path = os.path.join(directory, file)

                    # Split the markdown file into chunks
                    markdown_splits = self.split_markdown(markdown_path)

                    # Add the chunks to the overall list
                    all_splits.extend(markdown_splits)
                    print(f"Total chunks so far: {len(all_splits)}")  # Log the total number of chunks

            return all_splits
        except Exception as e:
            print(f"Error splitting markdown directory: {e}")
            return []


class Retriever:
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        
    def similarity_search_withscore(self, query, k=5):
        """
        Perform a similarity search on the ChromaDB index.
        """
        documents_with_scores = self.vectorstore.similarity_search_with_relevance_scores(query,k=k)

        if not documents_with_scores:
            print("No relevant context found.")
        
        documents = [doc.page_content for doc, _ in documents_with_scores]
        scores = [score for _, score in documents_with_scores]
        joined_documents = "\n".join(documents)
        
        return joined_documents, scores
        return "\n".join([f"{doc.page_content}" for doc ,score in documents_with_scores])
    


    def retrieve_multiquery_context(self, query: str, llm) -> str:
        """
        Retrieves context using the MultiQueryRetriever.
        """

        retriever = MultiQueryRetriever.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever())
        
        
            

        retreived_docs = retriever.invoke(query)

        if not retreived_docs:
            return "No relevant context found."
        return "\n".join([f"{doc.page_content}" for doc in retreived_docs])


    def _rerank_documents(self, query: str, documents: List[Tuple], k: int = 3) -> List[Tuple]:
        
        """Reranks documents using a cross-encoder model."""
        
        
        
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # Simple reranker

        doc_texts = [doc.page_content for doc  in documents]
        pairs = [(query, doc_text) for doc_text in doc_texts]
        rerank_scores = reranker.predict(pairs)

        reranked = sorted(
            zip(documents, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return [item[0] for item in reranked]
    
    def rerank_documents(self, query,  k=3):
        """
        Rerank the raw documents using the LLM.
        """
        raw_context = self.vectorstore.similarity_search(query, k=10)

        reranked_documents = self._rerank_documents(query,raw_context,k=k)
        
        return "\n".join([f"{doc.page_content}" for doc in reranked_documents])
    
   