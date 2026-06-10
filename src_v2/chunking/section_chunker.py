from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src_v2.models import (
    Chunk,
    FlattenedSection,
    Document
)


class SectionChunker:

    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=200,
        max_section_length=2500
    ):

        self.chunk_size = chunk_size

        self.chunk_overlap = chunk_overlap

        self.max_section_length = (
            max_section_length
        )

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        )

    def chunk(
        self,
        document: Document,
        sections: list[FlattenedSection]
    ) -> list[Chunk]:

        chunks = []

        chunk_counter = 0

        for section in sections:

            # Skip container sections
            if len(section.content.strip()) == 0:
                continue

            # Small section
            if (
                len(section.content)
                <= self.max_section_length
            ):

                chunks.append(
                    Chunk(
                        chunk_id=f"chunk_{chunk_counter}",
                        parent_doc_id=document.doc_id,
                        parent_section_id=section.section_id,
                        chunk_order=chunk_counter,
                        fragment_index=-1,
                        section_title=section.title,
                        path=section.path,
                        level=section.level,
                        chunk_type="section",
                        content=section.content
                    )
                )

                chunk_counter += 1

            # Large section
            else:

                # Parent chunk
                chunks.append(
                    Chunk(
                        chunk_id=f"chunk_{chunk_counter}",
                        parent_doc_id=document.doc_id,
                        parent_section_id=section.section_id,
                        chunk_order=chunk_counter,
                        fragment_index=-1,
                        section_title=section.title,
                        path=section.path,
                        level=section.level,
                        chunk_type="parent_section",
                        content=section.content
                    )
                )

                chunk_counter += 1

                fragments = self.splitter.split_text(
                    section.content
                )

                for idx, fragment in enumerate(
                    fragments
                ):

                    chunks.append(
                        Chunk(
                            chunk_id=f"chunk_{chunk_counter}",
                            parent_doc_id=document.doc_id,
                            parent_section_id=section.section_id,
                            chunk_order=chunk_counter,
                            fragment_index=idx,
                            section_title=section.title,
                            path=section.path,
                            level=section.level,
                            chunk_type="section_fragment",
                            content=fragment
                        )
                    )

                    chunk_counter += 1

        return chunks