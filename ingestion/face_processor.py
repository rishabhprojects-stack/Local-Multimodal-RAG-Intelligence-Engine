import insightface
import numpy as np
import os
import json
import uuid
from PIL import Image

FACES_FILE = "storage/faces.json"
UNKNOWN_FOLDER = "storage/unknown_faces"

class FaceProcessor:
    def __init__(self):
        os.makedirs(UNKNOWN_FOLDER, exist_ok=True)

        if not os.path.exists(FACES_FILE):
            with open(FACES_FILE, "w") as f:
                json.dump({}, f)

        # Load InsightFace model
        self.app = insightface.app.FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=-1)  # use -1 for CPU

    # ------------------------
    # Load known faces
    # ------------------------
    def load_known_faces(self):
        try:
            if not os.path.exists(FACES_FILE):
                return {}

            with open(FACES_FILE, "r") as f:
                content = f.read().strip()

                if not content:
                    return {}

                return json.loads(content)

        except json.JSONDecodeError:
            # If corrupted, reset file safely
            with open(FACES_FILE, "w") as f:
                json.dump({}, f)
            return {}

    # ------------------------
    # Detect and identify faces
    # ------------------------
    def detect_and_identify(self, image_path, threshold=0.6):
        image = np.array(Image.open(image_path).convert("RGB"))
        faces = self.app.get(image)

        known_faces = self.load_known_faces()
        matched_names = []

        for face in faces:
            embedding = face.embedding

            name = self.match_face(embedding, known_faces, threshold)

            if name:
                matched_names.append(name)
            else:
                self.save_unknown_face(image, face, embedding)

        return matched_names

    # ------------------------
    # Compare embeddings
    # ------------------------
    def match_face(self, embedding, known_faces, threshold):
        for name, saved_embedding in known_faces.items():
            saved_embedding = np.array(saved_embedding)

            similarity = self.cosine_similarity(embedding, saved_embedding)

            if similarity > threshold:
                return name
        return None

    # ------------------------
    # Cosine similarity
    # ------------------------
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    # ------------------------
    # Save unknown face
    # ------------------------
    def save_unknown_face(self, image, face, embedding):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        face_crop = image[y1:y2, x1:x2]

        face_id = str(uuid.uuid4())

        image_path = os.path.join(UNKNOWN_FOLDER, f"{face_id}.jpg")
        encoding_path = os.path.join(UNKNOWN_FOLDER, f"{face_id}.json")

        Image.fromarray(face_crop).save(image_path)

        with open(encoding_path, "w") as f:
            json.dump(embedding.tolist(), f)

    # ------------------------
    # Assign name to unknown face
    # ------------------------
    def assign_name(self, face_id, name):
        encoding_path = os.path.join(UNKNOWN_FOLDER, f"{face_id}.json")

        with open(encoding_path, "r") as f:
            embedding = json.load(f)

        known_faces = self.load_known_faces()
        known_faces[name] = embedding

        with open(FACES_FILE, "w") as f:
            json.dump(known_faces, f)
