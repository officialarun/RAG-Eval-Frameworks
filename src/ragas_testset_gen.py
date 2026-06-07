import sys
import os
import json
from unittest.mock import MagicMock
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()

from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import PyPDFLoader
from rag_pipeline import ask
from datasets import Dataset
from ragas import evaluate



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

loader = DirectoryLoader(
    os.path.join(BASE_DIR, "data/documents/"),
    glob="**/*.pdf",
    loader_cls=PyPDFLoader  # tell it to use PDF loader
)
docs = loader.load()
print("Number of docs:", len(docs))

for i, doc in enumerate(docs):
    print(i, doc.metadata)
docs=docs[:0]  # Limit to first 1 document for testing

generator_llm = LangchainLLMWrapper(
    ChatOllama(
        model="qwen3:8b"
    )
)
generator_embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
)
generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=generator_embeddings
)

dataset = generator.generate_with_langchain_docs(
    docs,
    testset_size=1,
    with_debugging_logs=True
)

df = dataset.to_pandas()
print(df.head())

output_path = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "ragas_testset.csv"
)

os.makedirs(
    os.path.dirname(output_path),
    exist_ok=True
)

df.to_csv(output_path, index=False)

print(df.columns)
print(df.iloc[0])