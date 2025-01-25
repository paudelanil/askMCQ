from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

# Step 1: Initialize SentenceTransformer embeddings with LangChain
model_name = "abhinand/MedEmbed-small-v0.1"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Step 2: Initialize Chroma vector store
persist_directory = "./chroma_langchain_data"

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=persist_directory
)

# Step 3: Add documents to the vector store
documents = [
    Document(page_content="The weather is lovely today."),
    Document(page_content="It's so sunny outside!"),
    Document(page_content="He drove to the stadium.")
]

# Embed and store the documents
vector_store.add_documents(documents)

# Persist the data for reuse
vector_store.persist()

# Step 4: Query the vector store
query = "It's bright and sunny."
results = vector_store.similarity_search(query, k=3)

# Display results
for result in results:
    print(f"Retrieved Document: {result.page_content}")
