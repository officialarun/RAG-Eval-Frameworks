import arxiv
from langchain_core.documents import Document


def load_arxiv_documents():

    queries = [
        "transformer",
        "large language model",
        "retrieval augmented generation",
        "attention mechanism",
        "vector database"
    ]

    documents = []

    for query in queries:

        search = arxiv.Search(
            query=query,
            max_results=10
        )
        client = arxiv.Client()

        for result in client.results(search):

            doc = Document(
                page_content=result.summary,
                metadata={
                    "source": "arxiv",
                    "title": result.title,
                    "authors": [author.name for author in result.authors],
                    "published": str(result.published),
                    "url": result.entry_id
                }
            )

            documents.append(doc)

    print(f"Loaded {len(documents)} Arxiv documents")

    return documents


if __name__ == "__main__":
    docs = load_arxiv_documents()

    print(docs[0].metadata)