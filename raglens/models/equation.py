from dataclasses import dataclass, field
@dataclass
class Equation:

    equation_id: str

    latex: str

    context: str

    metadata: dict = field(
        default_factory=dict
    )