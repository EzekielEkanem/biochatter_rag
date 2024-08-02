# Import libraries 
from biochatter.vectorstore_agent import VectorDatabaseAgentMilvus
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize an Hugging Face embedding model
model_name = "BAAI/bge-en-icl"
hf = HuggingFaceEmbeddings(model_name=model_name)

# establish a connection with the vector database
dbHost = VectorDatabaseAgentMilvus(embedding_func= hf)

# Load the Directory loader to read the documents
loader = DirectoryLoader("/home/ubuntu/biochatter/Bioc_contribution_downloads", 
                         glob="**/*.html", show_progress=True)
docs = loader.load()

# split documents
doc_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=[" ", ",", "\n"]
)
splitted_docs = doc_splitter.split_documents(docs)

# embed and store embeddings in the connected vector DB
doc_id = dbHost.store_embeddings(splitted_docs)

# Perform a semantic search and print it
results = dbHost.similarity_search(
    query="What must a Bioconductor package contain  before it can be accepted for submission",
    k=3,
)
print(results)