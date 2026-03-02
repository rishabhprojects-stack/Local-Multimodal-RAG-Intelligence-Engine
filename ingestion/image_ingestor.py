class ImageIngestor:
    def __init__(self, face_processor):
        self.face_processor = face_processor

    def process_image(self, image_path):
        # TODO: Replace with real caption model later
        caption = f"Image showing scene in {image_path}"

        # Detect and identify faces
        matched_faces = self.face_processor.detect_and_identify(image_path)

        return caption, matched_faces
