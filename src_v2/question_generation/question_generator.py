import json
import textwrap
import os

from dotenv import (
    load_dotenv
)

from openai import (
    OpenAI
)

from src_v2.models import (
    Chunk
)


class QuestionGenerator:

    def __init__(self):

        load_dotenv()

        self.client = OpenAI(
            api_key=os.getenv(
                "OPENAI_API_KEY"
            )
        )

    def build_prompt(
        self,
        chunk: Chunk
    ):

        prompt = f"""
        You are creating a high-quality evaluation dataset for a Retrieval-Augmented Generation (RAG) system.

        SECTION TITLE:
        {chunk.section_title}

        SECTION PATH:
        {chunk.path}

        CONTENT:
        {chunk.content}

        TASK:

        Determine whether the content contains enough meaningful information to create a useful evaluation question.

        RULES:

        1. Generate exactly ONE question.
        2. Generate exactly ONE reference answer.
        3. The answer must be directly supported by the content.
        4. Do not use outside knowledge.
        5. Do not reference figures, tables, sections, or information that is not present in the content.
        6. Prefer conceptual, factual, explanatory, or comparison questions.
        7. Avoid trivial questions.
        8. Avoid questions such as:
        - What is discussed in this section?
        - What does this section explain?
        9. Questions must be written as natural standalone questions.

        10. Do NOT use phrases such as:
        - According to the text
        - According to the content
        - Based on the passage
        - As stated above
        - In this section
        - In the document

        11. The question should read as if asked by a real user.

        If the content is unsuitable for evaluation question generation, return:

        {{
            "status": "skip",
            "reason": "insufficient_content"
        }}

        Otherwise return:

        {{
            "status": "success",
            "question": "...",
            "reference_answer": "..."
        }}

        Return ONLY valid JSON.
        """

        return (
            textwrap
            .dedent(prompt)
            .strip()
        )

    def parse_response(
        self,
        response_text: str
    ):

        start = (
            response_text.find("{")
        )

        end = (
            response_text.rfind("}")
            + 1
        )

        json_text = (
            response_text[start:end]
        )

        return json.loads(
            json_text
        )

    def generate(
        self,
        chunk: Chunk
    ):

        prompt = (
            self.build_prompt(
                chunk
            )
        )

        response = (
            self.client.responses.create(
                model="gpt-5-nano",
                input=prompt
            )
        )

        response_text = (
            response.output_text
        )

        return (
            self.parse_response(
                response_text
            )
        )