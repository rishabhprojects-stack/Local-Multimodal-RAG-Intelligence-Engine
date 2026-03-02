from fastembed import TextEmbedding
from config import EMBEDDING_MODEL

class Embedder:
    def __init__(self):
        self.model_name = EMBEDDING_MODEL 
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL)

    def embed(self, texts):
        return list(self.model.embed(texts))
