"""
Streamlit App for RF Model Training

A user-friendly interface to:
- Upload PDF documents progressively
- Generate synthetic queries
- Train RF model for RAG chunk scoring
- View training progress and metrics
- Export trained model

Run with: streamlit run app.py
"""

import os
import io
import json
import time
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np

import streamlit as st

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# HTTP for Gemini API
import httpx

# PDF extraction
from PyPDF2 import PdfReader

# ============================================
# Page Config
# ============================================

st.set_page_config(
    page_title="RF Training Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4f46e5, #0ea5e9);
    }
    .upload-box {
        border: 2px dashed #4f46e5;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background: rgba(79, 70, 229, 0.05);
    }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2d2d3d);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(79, 70, 229, 0.3);
    }
    .doc-item {
        background: rgba(255,255,255,0.05);
        padding: 8px 12px;
        border-radius: 6px;
        margin: 4px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Data Classes
# ============================================

@dataclass
class Document:
    id: str
    filename: str
    content: str
    num_pages: int
    num_chunks: int = 0


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_filename: str
    content: str
    chunk_index: int
    total_chunks: int
    token_estimate: int
    embedding: Optional[List[float]] = None


# ============================================
# Persistence - Save/Load from Disk
# ============================================

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

def save_to_disk(key: str, data):
    """Save data to disk cache."""
    try:
        filepath = CACHE_DIR / f"{key}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving {key}: {e}")

def load_from_disk(key: str, default=None):
    """Load data from disk cache."""
    try:
        filepath = CACHE_DIR / f"{key}.pkl"
        if filepath.exists():
            with open(filepath, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading {key}: {e}")
    return default

def save_all_state():
    """Save all session state to disk."""
    save_to_disk("documents", st.session_state.documents)
    save_to_disk("chunks", st.session_state.chunks)
    save_to_disk("embeddings_cache", st.session_state.embeddings_cache)
    save_to_disk("training_data", st.session_state.training_data)
    if st.session_state.model:
        save_to_disk("model", st.session_state.model)

def clear_all_cache():
    """Clear all cached data from disk."""
    for f in CACHE_DIR.glob("*.pkl"):
        f.unlink()


# ============================================
# Session State Initialization
# ============================================

def attach_embeddings_to_chunks():
    """Attach cached embeddings to chunk objects."""
    attached = 0
    for chunk in st.session_state.chunks:
        if chunk.id in st.session_state.embeddings_cache:
            chunk.embedding = st.session_state.embeddings_cache[chunk.id]
            attached += 1
    return attached

def init_session_state():
    """Initialize session state variables, loading from disk if available."""
    if "initialized" not in st.session_state:
        # First load - try to restore from disk
        st.session_state.documents = load_from_disk("documents", [])
        st.session_state.chunks = load_from_disk("chunks", [])
        st.session_state.embeddings_cache = load_from_disk("embeddings_cache", {})
        st.session_state.training_data = load_from_disk("training_data", [])
        st.session_state.model = load_from_disk("model", None)
        st.session_state.training_log = []
        st.session_state.is_training = False
        st.session_state.initialized = True
        
        # CRITICAL: Attach embeddings to chunks after loading from disk
        if st.session_state.chunks and st.session_state.embeddings_cache:
            attached = attach_embeddings_to_chunks()
            print(f"[Init] Attached {attached} embeddings to chunks")
        
        # Show restore message if data was loaded
        if st.session_state.documents:
            st.toast(f"✅ Restored {len(st.session_state.documents)} documents from previous session")


# ============================================
# PDF Processing
# ============================================

def extract_text_from_pdf(pdf_file) -> tuple[str, int]:
    """Extract text from uploaded PDF file."""
    try:
        reader = PdfReader(pdf_file)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts), len(reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return "", 0


def chunk_document(doc: Document, chunk_size: int = 1000) -> List[Chunk]:
    """Split document into chunks."""
    text = doc.content
    chunks = []
    
    paragraphs = text.split("\n\n")
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if current_chunk and len(current_chunk) + len(para) > chunk_size:
            chunk_id = f"{doc.id}_{chunk_index}"
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc.id,
                doc_filename=doc.filename,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                total_chunks=0,
                token_estimate=len(current_chunk.split())
            ))
            chunk_index += 1
            current_chunk = para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
    
    if current_chunk.strip():
        chunk_id = f"{doc.id}_{chunk_index}"
        chunks.append(Chunk(
            id=chunk_id,
            doc_id=doc.id,
            doc_filename=doc.filename,
            content=current_chunk.strip(),
            chunk_index=chunk_index,
            total_chunks=0,
            token_estimate=len(current_chunk.split())
        ))
    
    for chunk in chunks:
        chunk.total_chunks = len(chunks)
    
    return chunks


# ============================================
# Gemini API
# ============================================

class GeminiClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.request_count = 0
        self.last_request = 0
    
    def _rate_limit(self):
        """Rate limiting - 30 RPM."""
        now = time.time()
        if now - self.last_request < 2:  # Min 2 sec between requests
            time.sleep(2 - (now - self.last_request))
        self.last_request = time.time()
    
    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding."""
        self._rate_limit()
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
        
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(url, json={
                    "model": "models/text-embedding-004",
                    "content": {"parts": [{"text": text[:25000]}]}
                })
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("embedding", {}).get("values", [])
                return None
        except:
            return None
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text."""
        self._rate_limit()
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        
        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"maxOutputTokens": max_tokens}
                })
                
                if response.status_code == 200:
                    data = response.json()
                    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                    return parts[0].get("text", "") if parts else ""
                return ""
        except:
            return ""


# ============================================
# Feature Extraction
# ============================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def extract_features(query: str, chunk: Chunk, similarity: float) -> Dict[str, float]:
    """Extract features for a query-chunk pair."""
    position = chunk.chunk_index / max(chunk.total_chunks, 1)
    
    query_words = set(query.lower().split())
    chunk_words = set(chunk.content.lower().split())
    keyword_overlap = len(query_words & chunk_words) / max(len(query_words), 1)
    
    chunk_length = len(chunk.content)
    has_numbers = 1.0 if any(c.isdigit() for c in chunk.content) else 0.0
    has_bullets = 1.0 if any(x in chunk.content for x in ["•", "-", "*", "1.", "2."]) else 0.0
    sentence_count = chunk.content.count(".") + chunk.content.count("?")
    
    return {
        "similarity": similarity,
        "position": position,
        "position_early": 1.0 if position < 0.3 else 0.0,
        "position_middle": 1.0 if 0.3 <= position < 0.7 else 0.0,
        "position_late": 1.0 if position >= 0.7 else 0.0,
        "keyword_overlap": keyword_overlap,
        "chunk_length": chunk_length / 2000,
        "token_count": chunk.token_estimate / 500,
        "has_numbers": has_numbers,
        "has_bullets": has_bullets,
        "sentence_count": sentence_count / 20,
        "query_length": len(query.split()) / 20,
    }


# ============================================
# Training Pipeline
# ============================================

def run_training(api_key: str, num_queries: int, progress_callback):
    """Run the full training pipeline."""
    
    gemini = GeminiClient(api_key)
    chunks = st.session_state.chunks
    docs = st.session_state.documents
    
    if not chunks:
        st.error("No chunks available. Upload and process documents first.")
        return None
    
    if not docs:
        st.error("No documents available. Upload documents first.")
        return None
    
    total_steps = 4
    current_step = 0
    
    # Step 1: Generate embeddings for chunks not in cache
    progress_callback(current_step / total_steps, "Checking embeddings...")
    
    chunks_to_embed = [c for c in chunks if c.id not in st.session_state.embeddings_cache]
    
    st.info(f"📊 {len(chunks)} chunks total, {len(st.session_state.embeddings_cache)} in cache, {len(chunks_to_embed)} need embedding")
    
    if chunks_to_embed:
        embed_failures = 0
        for i, chunk in enumerate(chunks_to_embed):
            embedding = gemini.embed(chunk.content)
            if embedding:
                st.session_state.embeddings_cache[chunk.id] = embedding
            else:
                embed_failures += 1
            progress_callback(
                (current_step + (i + 1) / len(chunks_to_embed)) / total_steps,
                f"Embedding chunk {i+1}/{len(chunks_to_embed)}"
            )
        
        if embed_failures > 0:
            st.warning(f"⚠️ {embed_failures} embeddings failed to generate (API issues)")
    else:
        progress_callback(0.25, "All chunks already have embeddings ✓")
    
    # Apply ALL cached embeddings to chunk objects
    embedded_count = 0
    for chunk in chunks:
        if chunk.id in st.session_state.embeddings_cache:
            chunk.embedding = st.session_state.embeddings_cache[chunk.id]
            embedded_count += 1
    
    st.info(f"✅ Attached embeddings to {embedded_count}/{len(chunks)} chunks")
    
    # Validate we have enough embedded chunks
    embedded_chunks = [c for c in chunks if c.embedding]
    if len(embedded_chunks) < 10:
        st.error(f"Not enough chunks with embeddings ({len(embedded_chunks)}). Need at least 10.")
        return None
    
    # Save embeddings to disk (most expensive to regenerate)
    save_to_disk("embeddings_cache", st.session_state.embeddings_cache)
    
    current_step = 1
    
    # Step 2: Generate queries
    progress_callback(current_step / total_steps, "Generating synthetic queries...")
    
    queries_per_doc = max(1, num_queries // len(docs))
    all_queries = []
    
    st.info(f"📝 Generating ~{queries_per_doc} queries per document ({len(docs)} docs)")
    
    for i, doc in enumerate(docs):
        doc_preview = doc.content[:3000]
        prompt = f"""Based on this document, generate {queries_per_doc} diverse questions.

Document: {doc.filename}
---
{doc_preview}
---

Generate {queries_per_doc} questions (factual, conceptual, specific, broad).
Output ONLY questions, one per line:"""

        response = gemini.generate(prompt, 500)
        questions = [q.strip() for q in response.split("\n") if q.strip() and "?" in q]
        
        for q in questions[:queries_per_doc]:
            all_queries.append({"query": q, "doc": doc.filename})
        
        progress_callback(
            (current_step + (i + 1) / len(docs)) / total_steps,
            f"Generated queries for {doc.filename}"
        )
    
    if not all_queries:
        st.error("Failed to generate any queries. Check your API key.")
        return None
    
    st.info(f"✅ Generated {len(all_queries)} queries total")
    
    current_step = 2
    
    # Step 3: Collect training data
    progress_callback(current_step / total_steps, "Collecting training data...")
    
    training_data = []
    # Use the embedded_chunks we already validated above
    
    st.info(f"🔄 Processing {len(all_queries)} queries against {len(embedded_chunks)} chunks...")
    
    skipped_queries = 0
    for i, q_data in enumerate(all_queries):
        query = q_data["query"]
        
        # Embed query
        query_emb = gemini.embed(query)
        if not query_emb:
            skipped_queries += 1
            continue
        
        # Find similar chunks
        scored = [(c, cosine_similarity(query_emb, c.embedding)) 
                  for c in embedded_chunks if c.embedding]
        scored.sort(key=lambda x: -x[1])
        candidates = scored[:20]
        
        if len(candidates) < 5:
            skipped_queries += 1
            continue
        
        # Generate answer
        top_chunks = [c for c, _ in candidates[:5]]
        context = "\n\n---\n\n".join([c.content[:800] for c in top_chunks])
        answer = gemini.generate(f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:", 400)
        
        if not answer:
            skipped_queries += 1
            continue
        
        # Judge each chunk
        for chunk, sim in candidates[:10]:  # Judge top 10
            # LLM judges
            judge_prompt = f"""Was this chunk useful for answering the question?

Question: {query}
Answer: {answer[:500]}
Chunk: {chunk.content[:800]}

Respond with ONLY "yes" or "no":"""
            
            verdict = gemini.generate(judge_prompt, 10).strip().lower()
            label = 1 if "yes" in verdict else 0
            
            features = extract_features(query, chunk, sim)
            training_data.append({
                "features": features,
                "label": label,
                "query": query,
                "chunk_id": chunk.id
            })
        
        progress_callback(
            (current_step + (i + 1) / len(all_queries)) / total_steps,
            f"Processing query {i+1}/{len(all_queries)} ({len(training_data)} samples)"
        )
    
    if skipped_queries > 0:
        st.warning(f"⚠️ Skipped {skipped_queries} queries due to API issues or insufficient candidates")
    
    st.info(f"✅ Collected {len(training_data)} training samples")
    
    st.session_state.training_data = training_data
    current_step = 3
    
    # Step 4: Train RF
    progress_callback(current_step / total_steps, "Training Random Forest...")
    
    if len(training_data) < 50:
        st.error(f"Not enough training data ({len(training_data)} examples). Need at least 50.")
        return None
    
    feature_names = list(training_data[0]["features"].keys())
    X = np.array([[ex["features"][f] for f in feature_names] for ex in training_data])
    y = np.array([ex["label"] for ex in training_data])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    model_data = {
        "model": rf,
        "feature_names": feature_names,
        "accuracy": accuracy,
        "training_samples": len(training_data),
        "positive_rate": float(y.mean()),
        "feature_importance": dict(zip(feature_names, rf.feature_importances_.tolist()))
    }
    
    st.session_state.model = model_data
    
    # Save everything to disk
    save_all_state()
    
    progress_callback(1.0, "Training complete! ✅ Saved to disk")
    
    return model_data


# ============================================
# UI Components
# ============================================

def render_sidebar():
    """Render sidebar with settings and stats."""
    
    with st.sidebar:
        st.title("🧠 RF Training Studio")
        st.markdown("---")
        
        # API Key
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
            help="Your Gemini API key for embeddings and generation"
        )
        
        st.markdown("---")
        
        # Stats
        st.subheader("📊 Current Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.documents))
        with col2:
            st.metric("Chunks", len(st.session_state.chunks))
        
        # Count chunks that actually have embeddings attached
        chunks_with_embeddings = sum(1 for c in st.session_state.chunks if c.embedding)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Embedded", f"{chunks_with_embeddings}/{len(st.session_state.chunks)}")
        with col2:
            st.metric("Training Samples", len(st.session_state.training_data))
        
        # Sync button if there's a mismatch
        if chunks_with_embeddings < len(st.session_state.chunks) and st.session_state.embeddings_cache:
            if st.button("🔄 Sync Embeddings"):
                attached = attach_embeddings_to_chunks()
                st.toast(f"Attached {attached} embeddings")
                st.rerun()
        
        st.markdown("---")
        
        # Training settings
        st.subheader("⚙️ Training Settings")
        num_queries = st.slider("Queries to generate", 50, 300, 150, 10)
        chunk_size = st.slider("Chunk size (chars)", 500, 2000, 1000, 100)
        
        st.markdown("---")
        
        # Actions
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.documents = []
            st.session_state.chunks = []
            st.session_state.embeddings_cache = {}
            st.session_state.training_data = []
            st.session_state.model = None
            clear_all_cache()  # Also clear disk
            st.rerun()
        
        return api_key, num_queries, chunk_size


def render_upload_section(chunk_size: int):
    """Render document upload section."""
    
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF documents for training. You can add more anytime."
    )
    
    if uploaded_files:
        new_docs = 0
        existing_ids = {d.id for d in st.session_state.documents}
        
        for file in uploaded_files:
            doc_id = hashlib.md5(file.name.encode()).hexdigest()[:12]
            
            if doc_id not in existing_ids:
                content, num_pages = extract_text_from_pdf(file)
                
                if content.strip():
                    doc = Document(
                        id=doc_id,
                        filename=file.name,
                        content=content,
                        num_pages=num_pages
                    )
                    
                    # Chunk the document
                    doc_chunks = chunk_document(doc, chunk_size)
                    doc.num_chunks = len(doc_chunks)
                    
                    st.session_state.documents.append(doc)
                    st.session_state.chunks.extend(doc_chunks)
                    new_docs += 1
        
        if new_docs > 0:
            # Try to attach any cached embeddings to new chunks
            attach_embeddings_to_chunks()
            save_all_state()  # Persist to disk
            st.success(f"Added {new_docs} new document(s)! ✅ Saved to disk")
            st.rerun()
    
    # Show current documents
    if st.session_state.documents:
        st.subheader("📚 Uploaded Documents")
        
        for doc in st.session_state.documents:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            with col1:
                st.text(f"📄 {doc.filename}")
            with col2:
                st.text(f"{doc.num_pages} pages")
            with col3:
                st.text(f"{doc.num_chunks} chunks")
            with col4:
                if st.button("❌", key=f"del_{doc.id}"):
                    # Remove document and its chunks
                    st.session_state.documents = [d for d in st.session_state.documents if d.id != doc.id]
                    chunks_to_remove = [c.id for c in st.session_state.chunks if c.doc_id == doc.id]
                    st.session_state.chunks = [c for c in st.session_state.chunks if c.doc_id != doc.id]
                    
                    # Clean up stale embeddings from cache
                    for chunk_id in chunks_to_remove:
                        if chunk_id in st.session_state.embeddings_cache:
                            del st.session_state.embeddings_cache[chunk_id]
                    
                    save_all_state()  # Persist deletion
                    st.rerun()


def render_training_section(api_key: str, num_queries: int):
    """Render training section."""
    
    st.header("🚀 Train Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if not api_key:
            st.warning("⚠️ Enter your Gemini API key in the sidebar")
        elif len(st.session_state.documents) < 5:
            st.warning(f"⚠️ Upload at least 5 documents (currently: {len(st.session_state.documents)})")
        else:
            st.success(f"✅ Ready to train with {len(st.session_state.documents)} documents and {len(st.session_state.chunks)} chunks")
    
    with col2:
        can_train = api_key and len(st.session_state.documents) >= 5
        
        if st.button("🎯 Start Training", disabled=not can_train, type="primary"):
            st.session_state.is_training = True
    
    # Training progress
    if st.session_state.is_training:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(pct, text):
            progress_bar.progress(pct)
            status_text.text(text)
        
        with st.spinner("Training in progress..."):
            model_data = run_training(api_key, num_queries, update_progress)
        
        st.session_state.is_training = False
        
        if model_data:
            st.balloons()
            st.success("🎉 Training complete!")


def render_results_section():
    """Render model results section."""
    
    st.header("📈 Model Results")
    
    if not st.session_state.model:
        st.info("Train a model to see results here")
        return
    
    model_data = st.session_state.model
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{model_data['accuracy']:.1%}")
    with col2:
        st.metric("Training Samples", model_data['training_samples'])
    with col3:
        st.metric("Positive Rate", f"{model_data['positive_rate']:.1%}")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    
    importance = model_data['feature_importance']
    sorted_features = sorted(importance.items(), key=lambda x: -x[1])
    
    for name, imp in sorted_features:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.text(name)
        with col2:
            st.progress(imp)
    
    # Download model
    st.subheader("💾 Export Model")
    
    model_bytes = io.BytesIO()
    joblib.dump(model_data, model_bytes)
    model_bytes.seek(0)
    
    st.download_button(
        label="📥 Download rf_model.pkl",
        data=model_bytes,
        file_name="rf_model.pkl",
        mime="application/octet-stream"
    )


# ============================================
# Main App
# ============================================

def main():
    init_session_state()
    
    api_key, num_queries, chunk_size = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📁 Documents", "🚀 Training", "📈 Results"])
    
    with tab1:
        render_upload_section(chunk_size)
    
    with tab2:
        render_training_section(api_key, num_queries)
    
    with tab3:
        render_results_section()


if __name__ == "__main__":
    main()

