import os
from ingestion.registry import FileRegistry
from ingestion.pdf_ingestor import PDFIngestor
from ingestion.image_ingestor import ImageIngestor

DATA_FOLDER = "data"

class DataScanner:
    def __init__(self, vectorstore, embedder, face_processor):
        self.registry = FileRegistry()
        self.pdf_ingestor = PDFIngestor()

        self.vectorstore = vectorstore
        self.embedder = embedder
        self.face_processor = face_processor
        self.image_ingestor = ImageIngestor(face_processor)

    def scan(self):
        for root, _, files in os.walk(DATA_FOLDER):
            for file in files:
                path = os.path.join(root, file)

                if self.registry.is_processed(path):
                    continue

                if file.lower().endswith(".pdf"):
                    self.process_pdf(path)

                elif file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.process_image(path)

                self.registry.mark_processed(path)

    def process_pdf(self, path):
        chunks = self.pdf_ingestor.extract(path)
        embeddings = self.embedder.embed(chunks)

        payloads = [
            {
                "text": chunk,
                "source": path,
                "type": "doc",
                "folder": "docs"
            }
            for chunk in chunks
        ]

        self.vectorstore.add(
            "collection_docs",
            embeddings,
            payloads
        )

    def process_image(self, path):
        caption, matched_faces = self.image_ingestor.process_image(path)

        embedding = self.embedder.embed([caption])[0]

        payload = {
            "text": caption,
            "source": path,
            "type": "image",
            "folder": "images",
            "faces": matched_faces
        }

        self.vectorstore.add(
            "collection_images",
            [embedding],
            [payload]
        )
