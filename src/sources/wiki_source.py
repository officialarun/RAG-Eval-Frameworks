from wikipediaapi import Wikipedia
from langchain_core.documents import Document

wiki = Wikipedia(
    user_agent="rag-evaluation-framework",
    language="en"
)

topics = [
    "Transformer_(deep_learning_architecture)",
    "Attention_(machine_learning)",
    "Large_language_model",
    "Artificial_neural_network",
    "Deep_learning",
    "Machine_learning",
    "Prompt_engineering",
    "Vector_database",
    "LangChain",
    "BERT_(language_model)",
    "GPT-3",
    "GPT-4",
    "Llama_(language_model)",
    "Word_embedding",
    "Sentence_embedding",
    "Semantic_search",
    "Information_retrieval",
    "Knowledge_graph",
    "Fine-tuning_(deep_learning)",
    "Reinforcement_learning",
]

def load_wikipedia_documents():

    documents = []

    for topic in topics:

        page = wiki.page(topic)

        if not page.exists():
            continue

        doc = Document(
            page_content=page.text,
            metadata={
                "source": "wikipedia",
                "title": page.title,
                "url": page.fullurl
            }
        )

        documents.append(doc)

    print(f"Loaded {len(documents)} Wikipedia documents")

    return documents

if __name__ == "__main__":
    docs = load_wikipedia_documents()

    print(docs[0].metadata)
    print(len(docs))