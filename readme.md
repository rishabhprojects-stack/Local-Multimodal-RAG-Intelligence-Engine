```markdown
# CortexRAG  
### A Local Multimodal Retrieval-Augmented Intelligence System

CortexRAG is a fully local, multimodal AI platform that combines:

- 📄 Document Retrieval (RAG)
- 🖼 Hybrid Image Search (Text + Vector)
- 🙂 Face Identification & Labeling
- 🧠 Local LLM Inference (Ollama)
- 📊 System Monitoring Dashboard
- 🗄 Vector Database (Qdrant)

The system runs entirely locally and does not rely on external APIs.

---

## 🚀 Features

### 📄 Document RAG
- PDF ingestion
- Smart chunking
- Embedding with FastEmbed
- Vector storage in Qdrant
- Context-grounded LLM responses

### 🖼 Hybrid Image Search
- Text-to-image retrieval
- Vector similarity matching
- Image metadata display

### 🙂 Face Identification
- Detect and store unknown faces
- Assign identities manually
- Persist face embeddings

### 📊 System Dashboard
- Total processed files
- Vector collection statistics
- Embedding model info
- LLM model info

---

## 🏗 Project Architecture

```

Data → Embedding → Vector Store → Retrieval → LLM → Streamlit UI

```

### Folder Structure

```

phase 3/
│
├── app.py
│
├── ingestion/
│   ├── scanner.py
│   ├── registry.py
│   ├── pdf_ingestor.py
│   ├── image_ingestor.py
│   └── face_processor.py
│
├── embeddings/
│   └── embedder.py
│
├── vectorstore/
│   └── qdrant_client.py
│
├── inference/
│   ├── llm.py
│   ├── doc_rag.py
│   └── hybrid_search.py
│
├── storage/
│   ├── file_registry.db
│   └── unknown_faces/
│
└── config.py

````

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>
cd project
````

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🧠 Install and Run Ollama

Install Ollama:

```bash
brew install ollama
```

Start Ollama:

```bash
ollama serve
```

Pull a model (recommended):

```bash
ollama pull llama3.2:8b
```

---

## 🗄 Run Qdrant

Option 1: Docker (recommended)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

Option 2: In-memory mode (development only)

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

App will be available at:

```
http://localhost:8501
```

---

## 🔍 How RAG Works

1. Documents are ingested and chunked.
2. Chunks are embedded using FastEmbed.
3. Embeddings are stored in Qdrant.
4. User query is embedded.
5. Top-K similar chunks are retrieved.
6. Context is passed to the local LLM.
7. Model generates grounded response.

---

## 🧠 Models Used

* Embeddings: FastEmbed (`BAAI/bge-small-en-v1.5`)
* LLM: LLaMA 3.2 via Ollama
* Vector DB: Qdrant

---

## 📌 Requirements

* Python 3.10+
* Docker (optional but recommended)
* Ollama installed
* Apple Silicon supported

---

## 🔐 Local-First Architecture

* No external API calls
* No OpenAI dependency
* All embeddings and inference are local
* Data never leaves your machine

---

## 📈 Future Improvements

* Similarity threshold filtering
* Hybrid BM25 + Vector search
* Response citation enforcement
* Query logging analytics
* Streaming token output
* Re-ranking layer

---

## 👨‍💻 Author

Harsh Tripathi
Multimodal AI Systems Engineer

---

## 📄 License

MIT License

```
