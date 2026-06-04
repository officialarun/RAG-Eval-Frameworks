from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# -------------------------
# LLM
# -------------------------

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0
)


# -------------------------
# Embeddings
# -------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -------------------------
# Vector Store
# -------------------------

vector_store = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embeddings
)


# -------------------------
# Retriever
# -------------------------

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)


# -------------------------
# Main RAG Function
# -------------------------

def ask(question):

    # Retrieve relevant chunks
    docs = retriever.invoke(question)
    print("\nRETRIEVED CONTEXT:\n")

    for i, doc in enumerate(docs):
        print("=" * 60)
        print(f"Chunk {i+1}")
        print(doc.metadata)
        print(doc.page_content[:1000])

    # Store chunk ids for evaluation later
    retrieved_chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc in docs
    ]

    # Build context
    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    # Prompt
    prompt = f"""
    You are answering questions for a RAG evaluation system.

    Rules:
    1. Use ONLY the provided context.
    2. If the answer is not present in the context, say:
        "I don't know based on the provided context."
    3. Keep answers under 4 sentences.
    4. Do not add external knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Generate answer
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "retrieved_docs": docs,
        "retrieved_chunk_ids": retrieved_chunk_ids
    }


# -------------------------
# Testing
# -------------------------

if __name__ == "__main__":

    question = "What is self attention?"

    result = ask(question)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(result["answer"])

    print("\nRETRIEVED CHUNK IDS:")
    print(result["retrieved_chunk_ids"])

    print("\nRETRIEVED SOURCES:")

    for doc in result["retrieved_docs"]:

        print(
            f"{doc.metadata.get('title')} | "
            f"Chunk {doc.metadata.get('chunk_id')}"
        )