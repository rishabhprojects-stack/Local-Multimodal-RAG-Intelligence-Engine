import streamlit as st
import os
from ingestion.scanner import DataScanner
from ingestion.face_processor import FaceProcessor
from inference.hybrid_search import HybridImageSearch
from inference.doc_rag import DocumentRAG

st.set_page_config(layout="wide")

tabs = st.tabs(["Chat", "Face Identification", "Dashboard"])



from embeddings.embedder import Embedder
from vectorstore.qdrant_client import VectorStore
from inference.llm import LocalLLM

# ---------- INITIALIZATION (ONLY ONCE) ----------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = VectorStore()

if "embedder" not in st.session_state:
    st.session_state.embedder = Embedder()

if "face_processor" not in st.session_state:
    st.session_state.face_processor = FaceProcessor()

if "llm" not in st.session_state:
    st.session_state.llm = LocalLLM()

if "scanner" not in st.session_state:
    st.session_state.scanner = DataScanner(
        st.session_state.vectorstore,
        st.session_state.embedder,
        st.session_state.face_processor
    )

if "hybrid_search" not in st.session_state:
    st.session_state.hybrid_search = HybridImageSearch(
        st.session_state.embedder,
        st.session_state.vectorstore,
        st.session_state.face_processor
    )

if "doc_rag" not in st.session_state:
    st.session_state.doc_rag = DocumentRAG(
        st.session_state.embedder,
        st.session_state.vectorstore,
        st.session_state.llm
    )

scanner = st.session_state.scanner
hybrid_search = st.session_state.hybrid_search
doc_rag = st.session_state.doc_rag
face_processor = st.session_state.face_processor

# Run scan only once
#if "scanned" not in st.session_state:
#    scanner.scan()
#    st.session_state.scanned = True

# ---------------- CHAT TAB ----------------
with tabs[0]:
    st.title("Multimodal RAG Chat")

    mode = st.radio(
    "Select Mode",
    ["Search Images (Hybrid)", "Ask Documents (RAG)"]
    )

    query = st.text_input("Enter your query")

    if st.button("Search"):

        if mode == "Search Images (Hybrid)":
            results = hybrid_search.search(query)

            for score, payload in results:
                st.image(payload["source"], width=250)
                st.write(payload["text"])
                st.write(f"Score: {score}")

        else:
            answer, images = doc_rag.query(query)

            st.write(answer)

            for img in images:
                st.image(img, width=250)

# ---------------- FACE TAB ----------------
with tabs[1]:
    st.title("Identify Unknown Faces")

    unknown_folder = "storage/unknown_faces"
    files = [f for f in os.listdir(unknown_folder) if f.endswith(".jpg")]

    for file in files:
        face_id = file.replace(".jpg", "")
        image_path = os.path.join(unknown_folder, file)

        st.image(image_path, width=150)

        name = st.text_input(f"Name for {face_id}", key=face_id)

        if st.button(f"Assign {face_id}", key=f"btn_{face_id}"):
            face_processor.assign_name(face_id, name)
            st.success(f"Assigned {name} successfully!")


# ---------------- DASHBOARD TAB ----------------
# ---------------- DASHBOARD TAB ----------------
with tabs[2]:
    st.title("System Dashboard")

    col1, col2 = st.columns(2)

    # ---------------- FILE STATS ----------------
    with col1:
        st.subheader("Data Stats")

        total_files = len(
            scanner.registry.conn.execute("SELECT * FROM files").fetchall()
        )

        st.metric("Processed Files", total_files)

        # Count vectors in Qdrant
        try:
            collections = st.session_state.vectorstore.client.get_collections()

            if collections.collections:
                for col in collections.collections:
                    info = st.session_state.vectorstore.client.get_collection(col.name)

                    st.markdown(f"###{col.name}")
                    st.metric("Total Vectors", info.points_count)
                    st.write("Vector Dimension:", info.config.params.vectors.size)
                    st.divider()
            else:
                st.warning("No collections found.")

        except Exception as e:
            st.error(f"Vector DB error: {e}")

    # ---------------- SYSTEM INFO ----------------
    with col2:
        st.subheader("System Info")

        st.write("LLM Model:", st.session_state.llm.model)
        st.write("Embedding Model:", st.session_state.embedder.model_name)

        st.write("Storage Path:", "storage/")
