from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src_v2.chunking.structure_preserver import (
    StructurePreserver
)

from src_v2.models import (
    Chunk,
    FlattenedSection,
    Document
)

import re


class SectionChunker:

    def __init__(
        self,
        chunk_size=1200,
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

        self.preserver = (
            StructurePreserver()
        )
        self.safety_splitter = (
            RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=200
            )
        )

        self.max_embeddable_length = 5000

    def chunk(
        self,
        document: Document,
        sections: list[FlattenedSection]
    ) -> list[Chunk]:

        chunks = []

        chunk_counter = 0

        for section in sections:

            # ----------------------------------
            # Skip empty sections
            # ----------------------------------
            if len(
                section.content.strip()
            ) == 0:
                continue

            # ----------------------------------
            # Small section
            # ----------------------------------
            if (
                len(section.content)
                <= self.max_section_length
            ):

                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}"f"_{section.section_id}"f"_{chunk_counter}",
                        parent_doc_id=document.doc_id,
                        parent_section_id=section.section_id,
                        chunk_order=chunk_counter,
                        fragment_index=-1,
                        section_title=section.section_title,
                        path=section.path,
                        level=section.level,
                        chunk_type="parent_section",
                        content=section.content,
                        metadata={}
                    )
                )

                chunk_counter += 1

            # ----------------------------------
            # Large section
            # ----------------------------------
            else:

                # Parent section chunk
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}"f"_{section.section_id}"f"_{chunk_counter}",
                        parent_doc_id=document.doc_id,
                        parent_section_id=section.section_id,
                        chunk_order=chunk_counter,
                        fragment_index=-1,
                        section_title=section.section_title,
                        path=section.path,
                        level=section.level,
                        chunk_type="parent_section",
                        content=section.content,
                        metadata={}
                    )
                )

                chunk_counter += 1

                protected_text, mapping = (
                    self.preserver.protect(
                        section.content
                    )
                )

                # ----------------------------------
                # Create table fragment chunks
                # directly
                # ----------------------------------

                table_placeholders = re.findall(
                    r"__TABLE_\d+_FRAGMENT_\d+__",
                    protected_text
                )

                for placeholder in (
                    table_placeholders
                ):

                    table_info = (
                        mapping[
                            placeholder
                        ]
                    )

                    chunks.append(
                        Chunk(
                            chunk_id=f"{document.doc_id}"f"_{section.section_id}"f"_{chunk_counter}",
                            parent_doc_id=document.doc_id,
                            parent_section_id=section.section_id,
                            chunk_order=chunk_counter,
                            fragment_index=(
                                table_info[
                                    "table_fragment_index"
                                ]
                            ),
                            section_title=section.section_title,
                            path=section.path,
                            level=section.level,
                            chunk_type="table_fragment",
                            content=(
                                table_info[
                                    "content"
                                ]
                            ),
                            metadata={
                                "table_id":
                                    table_info[
                                        "table_id"
                                    ],

                                "table_fragment_index":
                                    table_info[
                                        "table_fragment_index"
                                    ],

                                "table_fragment_count":
                                    table_info[
                                        "table_fragment_count"
                                    ]
                            }
                        )
                    )

                    chunk_counter += 1

                # ----------------------------------
                # Remove table placeholders
                # before recursive splitting
                # ----------------------------------

                text_without_tables = (
                    protected_text
                )

                for placeholder in (
                    table_placeholders
                ):

                    text_without_tables = (
                        text_without_tables.replace(
                            placeholder,
                            ""
                        )
                    )

                # ----------------------------------
                # Split remaining text
                # ----------------------------------

                fragments = (
                    self.splitter.split_text(
                        text_without_tables
                    )
                )

                for idx, fragment in enumerate(
                    fragments
                ):

                    restored_fragment = (
                        self.preserver.restore(
                            fragment,
                            mapping
                        )
                    )

                    if (
                        len(
                            restored_fragment.strip()
                        ) == 0
                    ):
                        continue

                    # ----------------------------------
                    # Safety split for embedding limits
                    # ----------------------------------

                    if (
                        len(restored_fragment)
                        > self.max_embeddable_length
                    ):

                        safety_fragments = (
                            self.safety_splitter.split_text(
                                restored_fragment
                            )
                        )

                        for safety_idx, safety_fragment in enumerate(
                            safety_fragments
                        ):

                            chunks.append(
                                Chunk(
                                    chunk_id=f"{document.doc_id}"f"_{section.section_id}"f"_{chunk_counter}",
                                    parent_doc_id=document.doc_id,
                                    parent_section_id=section.section_id,
                                    chunk_order=chunk_counter,
                                    fragment_index=safety_idx,
                                    section_title=section.section_title,
                                    path=section.path,
                                    level=section.level,
                                    chunk_type="section_fragment",
                                    content=safety_fragment,
                                    metadata={}
                                )
                            )

                            chunk_counter += 1

                    else:

                        chunks.append(
                            Chunk(
                                chunk_id=f"{document.doc_id}"f"_{section.section_id}"f"_{chunk_counter}",
                                parent_doc_id=document.doc_id,
                                parent_section_id=section.section_id,
                                chunk_order=chunk_counter,
                                fragment_index=idx,
                                section_title=section.section_title,
                                path=section.path,
                                level=section.level,
                                chunk_type="section_fragment",
                                content=restored_fragment,
                                metadata={}
                            )
                        )

                        chunk_counter += 1

        return chunks