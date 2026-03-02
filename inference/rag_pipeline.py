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
        You are a strict retrieval-based assistant.

        You must answer ONLY using the information provided in the context below.

        If the answer is not explicitly found in the context,
        respond with:

        "I could not find the answer in the provided documents."

        Do NOT use prior knowledge.
        Do NOT guess.
        Do NOT use external information.

        ---------------------
        CONTEXT:
        {context}
        ---------------------

        QUESTION:
        {question}

        Answer:
        """
