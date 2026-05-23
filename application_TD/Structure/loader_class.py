from docx import Document as DocxDocument
from typing import List, Optional
import io
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class DOCXLoader(BaseLoader):
    """Loads a DOCX file (or stream) into a list of documents with metadata."""


    def __init__(self, file_content: Optional[bytes] = None): #constructor for the DOCXLoader
        """
        Initialize the loader with file content as bytes.
        :param file_content: Raw DOCX content as bytes (for API responses, etc.)
        """
        if file_content is not None:
            self.file_stream = io.BytesIO(file_content) #This creates an in-memory binary stream from the bytes, which many file-processing libraries can work with as if it were a file on disk.
            self.docx_document = DocxDocument(self.file_stream)
        else:
            raise ValueError("file_content must be provided.")

    def load(self) -> List[Document]:
        """
        Load the DOCX and extract paragraphs as Document objects with metadata.
        :return: A list of Document objects.
        """
        documents = []
        metadata = {
            "source": " ",  # Source will be set dynamically from caller
            'title': ' ',
            "page": 1
        }

        # Iterate over each paragraph in the DOCX
        for paragraph in self.docx_document.paragraphs:
            text = paragraph.text.strip()

            if text:  # Ignore empty paragraphs
                document = Document(page_content=text, metadata=metadata)
                documents.append(document)

        return documents