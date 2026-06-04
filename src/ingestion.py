from sources.wiki_source import load_wikipedia_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_chunks():

    documents = load_wikipedia_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks

if __name__ == "__main__":
    chunks = load_chunks()

    print(f"Chunks : {len(chunks)}")