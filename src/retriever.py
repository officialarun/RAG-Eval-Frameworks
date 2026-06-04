from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 8, "fetch_k": 30},
)

query = "What is self attention?"

results = retriever.invoke(query)

print(f"\nQuery: {query}")

for i, doc in enumerate(results):

    print("\n" + "=" * 60)
    print(f"Result {i+1}")
    print("=" * 60)

    print(doc.metadata)

    print(doc.page_content[:500])