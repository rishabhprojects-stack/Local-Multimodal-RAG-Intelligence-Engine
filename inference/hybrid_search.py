import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue

class HybridImageSearch:
    def __init__(self, embedder, vectorstore, face_processor):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.face_processor = face_processor

    def search(self, query):
        known_faces = self.face_processor.load_known_faces()
        matched_name = None

        # detect if query contains a known face name
        for name in known_faces.keys():
            if name.lower() in query.lower():
                matched_name = name
                break

        # remove name from query for caption embedding
        clean_query = query.replace(matched_name, "") if matched_name else query

        query_vector = self.embedder.embed([clean_query])[0]

        image_filter = Filter(
            must=[
                FieldCondition(
                    key="folder",
                    match=MatchValue(value="images")
                )
            ]
        )

        results = self.vectorstore.search(
            "collection_images",
            query_vector,
            limit=10,
            filter=image_filter
        )

        ranked_results = []

        for res in results:
            caption_score = res.score

            face_score = 0
            if matched_name and matched_name in res.payload.get("faces", []):
                face_score = 1

            hybrid_score = 0.6 * caption_score + 0.4 * face_score
            ranked_results.append((hybrid_score, res.payload))

        ranked_results.sort(reverse=True, key=lambda x: x[0])

        return ranked_results[:5]
