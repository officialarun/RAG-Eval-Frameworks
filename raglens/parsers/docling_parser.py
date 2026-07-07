
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption
)

from docling.datamodel.base_models import (
    InputFormat
)

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions
)


class DoclingParser:

    def __init__(self):

        options = PdfPipelineOptions()

        # important
        options.do_formula_enrichment = True

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=options
                )
            }
        )

    def parse(
        self,
        source: str
    ) -> str:

        result = self.converter.convert(
            source
        )

        return result.document.export_to_markdown()