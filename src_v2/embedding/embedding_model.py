from dataclasses import dataclass


@dataclass
class EmbeddingModel:

    model_name: str

    dimension: int