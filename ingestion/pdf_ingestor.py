from pypdf import PdfReader
from utils.chunking import chunk_text

class PDFIngestor:
    def extract(self, file_path):
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        return chunk_text(text)
