from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import chromadb
import os

# Set your OpenAI API key
with open("openai_api_key.txt", "r") as f:
    os.environ["OPENAI_API_KEY"] = f.read()
# os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

def load_and_process_pdfs(pdf_directory, persist_directory="./chroma_db", chunk_size=1000, chunk_overlap=200):
    """
    Load PDFs from a directory, split them into chunks, create embeddings,
    and store them in a Chroma database.
    
    Args:
        pdf_directory (str): Path to directory containing PDFs
        persist_directory (str): Path where Chroma DB will store its data
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        Chroma: Vector store for document retrieval
    """
    # Initialize PDF loader
    loader = DirectoryLoader(
        pdf_directory,
        glob="**/*.pdf",
        show_progress=True
    )
    
    # Load documents
    print("Loading PDF documents...")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    
    # Split documents into chunks
    print("Splitting documents into chunks...")
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} text chunks")
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",  # Using the latest embedding model
        chunk_size=500  # Process 500 texts at a time
    )
    
    # Create Chroma client
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Create or get collection
    collection_name = "sample_docs"
    
    # Create vector store
    print("Creating Chroma vector store...")
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory,
        client=client,
        collection_name=collection_name
    )
    
    # Persist the database
    vector_store.persist()
    print("Vector store created and persisted successfully")
    
    return vector_store

def search_documents(vector_store, query, k=5):
    """
    Search for relevant documents using similarity search.
    
    Args:
        vector_store (Chroma): Chroma vector store containing document embeddings
        query (str): Search query
        k (int): Number of results to return
    
    Returns:
        list: List of relevant documents with scores
    """
    # Perform similarity search
    results = vector_store.similarity_search_with_score(query, k=k)
    return results

def load_existing_db(persist_directory="./chroma_db"):
    """
    Load an existing Chroma database.
    
    Args:
        persist_directory (str): Path to the Chroma DB directory
    
    Returns:
        Chroma: Vector store for document retrieval
    """
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=1000
    )
    
    # Create Chroma client
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Load existing vector store
    vector_store = Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name="pdf_documents",
        persist_directory=persist_directory
    )
    
    return vector_store

def main():
    # Example usage
    pdf_dir = "./sample_pdfs"
    persist_dir = "./chroma_db"
    
    # Check if database already exists
    if os.path.exists(persist_dir):
        print("Loading existing Chroma database...")
        vector_store = load_existing_db(persist_dir)
    else:
        # Load and process documents
        vector_store = load_and_process_pdfs(pdf_dir, persist_dir)
    
    # Example search
    query = "What is machine learning?"
    results = search_documents(vector_store, query)
    
    # Print results
    print("\nSearch Results:")
    for doc, score in results:
        print(f"\nScore: {score}")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()