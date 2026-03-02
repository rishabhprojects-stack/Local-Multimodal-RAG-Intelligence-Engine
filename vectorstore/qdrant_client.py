from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import os
import uuid

class VectorStore:
    def __init__(self, dim=384):

        # Ensure storage folder exists
        os.makedirs("storage/qdrant", exist_ok=True)

        # qdrant client
        self.client = QdrantClient(
            path="storage/qdrant"
        )

        self._create_collections(dim)

    def _create_collections(self, dim):

        existing_collections = [
            c.name for c in self.client.get_collections().collections
        ]

        # Only created if not already existing (prevents wiping data)
        if "collection_docs" not in existing_collections:
            self.client.create_collection(
                collection_name="collection_docs",
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

        if "collection_images" not in existing_collections:
            self.client.create_collection(
                collection_name="collection_images",
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

    def add(self, collection, embeddings, payloads):

        points = [
            PointStruct(
                id=str(uuid.uuid4()),   # unique ID
                vector=vector,
                payload=payload,
            )
            for vector, payload in zip(embeddings, payloads)
        ]

        self.client.upsert(
            collection_name=collection,
            points=points
        )

    def search(self, collection, vector, limit=5, filter=None):
        return self.client.search(
            collection_name=collection,
            query_vector=vector,
            limit=limit,
            query_filter=filter
        )
