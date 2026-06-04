from ingestion import load_chunks
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def create_vector_store():

    print("Loading chunks...")

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")

    print("Loading embeddings model...")

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    print("Creating vector store...")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="../chroma_db"
    )

    print("Vector store created")

    return vector_store


if __name__ == "__main__":
    create_vector_store()