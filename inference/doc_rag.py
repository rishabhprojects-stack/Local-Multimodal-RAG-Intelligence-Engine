from qdrant_client.models import Filter, FieldCondition, MatchValue

class DocumentRAG:
    def __init__(self, embedder, vectorstore, llm):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm

    def query(self, question):
        vector = self.embedder.embed([question])[0]

        doc_filter = Filter(
            must=[
                FieldCondition(
                    key="folder",
                    match=MatchValue(value="docs")
                )
            ]
        )

        results = self.vectorstore.search(
            "collection_docs",
            vector,
            limit=5,
            filter=doc_filter
        )

        context = "\n".join([r.payload["text"] for r in results])

        prompt = f"""
        Use only this context to answer.

        {context}

        Question: {question}
        """

        answer = self.llm.generate(prompt)

        # retrieve related images
        image_results = self.vectorstore.search(
            "collection_images",
            vector,
            limit=3
        )

        images = [r.payload["source"] for r in image_results]

        return answer, images
