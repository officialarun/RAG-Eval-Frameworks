from dataclasses import dataclass, field

@dataclass
class QuestionSample:

    sample_id: str

    document_title: str
    
    question: str

    reference_answer: str

    supporting_context: str

    relevant_sections: list[str] = field(
        default_factory=list
    )

    metadata: dict = field(
        default_factory=dict
    )