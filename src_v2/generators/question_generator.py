import json
import textwrap
from typing import List
from langchain_ollama import ChatOllama
from src_v2.models import FlattenedSection, QuestionSample


class QuestionGenerator:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen3:8b",
            temperature=0,
        )

    def build_prompt(self, section: FlattenedSection) -> str:
        prompt = f"""
        You are creating a high-quality evaluation dataset for a Retrieval-Augmented Generation (RAG) system.

        SECTION PATH:
        {section.path}

        SECTION CONTENT:
        {section.content}

        Instructions:

        1. Generate exactly 3 question-answer pairs whenever the section contains sufficient information.
        2. Generate fewer only if the section genuinely lacks content.
        3. Each question must be answerable directly from the provided section.
        4. Avoid trivial wording such as:
        - What is discussed in this section?
        - What does this section explain?
        5. Prefer conceptual, factual, or explanatory questions.
        6. Generate a concise reference answer for each question.
        7. Do not invent information not present in the section.
        8. If the section lacks enough information, return an empty list.

        Return ONLY valid JSON.

        [
            {{
                "question": "...",
                "reference_answer": "..."
            }},
            {{
                "question": "...",
                "reference_answer": "..."
            }},
            {{
                "question": "...",
                "reference_answer": "..."
            }}
        ]
        """
        return textwrap.dedent(prompt).strip()

    def generate(self, section: FlattenedSection) -> List[QuestionSample]:
        raw_response = self.generate_raw(
            section
        )
        try:
            data = self.parse_response(
                raw_response
            )
        except Exception:
            print("\n" + "=" * 80)
            print(section.path)
            print("=" * 80)

            print(raw_response)
            raise

        samples = []
        for idx, item in enumerate(data):
            sample = QuestionSample(
                sample_id=f"{section.section_id}_{idx}",

                question=item["question"],

                reference_answer=item[
                    "reference_answer"
                ],

                supporting_context=
                    section.content,

                relevant_sections=[
                    section.path
                ]
                document_title=
                    section.document_title
            )
            samples.append(
            sample
            )

        return samples
    

    def generate_raw(self, section: FlattenedSection) -> str:
        prompt = self.build_prompt(section)
        response = self.llm.invoke(prompt)
        return response.content

    def parse_response(self, response):

        import re

        fixed = re.sub(
            r'"+\s*question\s*"+',
            '"question"',
            response
        )

        if fixed != response:

            print("JSON repaired")

        return json.loads(fixed)
