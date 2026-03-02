class RAGPipeline:
    def __init__(self, embedder, vectorstore, llm):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm

    def query(self, question):
        query_vector = self.embedder.embed([question])[0]
        results = self.vectorstore.search(query_vector)

        context = "\n".join(
            [res.payload["text"] for res in results]
        )

        print("RETRIEVED RESULTS:")
        for res in results:
            print(res.payload["text"][:200])

        prompt = f"""
        Use ONLY the following context to answer.

        Context:
        {context}

        Question:
        {question}
        """

        return self.llm.generate(prompt)
