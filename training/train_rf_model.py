"""
RF Model Training Script for RAG Chunk Scoring

This script:
1. Loads PDFs from training/docs/
2. Extracts text and chunks them
3. Generates embeddings using Gemini
4. Uses LLM to generate synthetic queries
5. For each query, retrieves chunks and judges utility
6. Trains Random Forest on collected data
7. Exports model to training/models/rf_model.pkl

Usage:
    python train_rf_model.py --docs_folder ./docs --num_queries 150
"""

import os
import io
import json
import random
import pickle
import hashlib
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# HTTP for Gemini API
import httpx

# PDF extraction
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("Installing PyPDF2...")
    os.system("pip install PyPDF2")
    from PyPDF2 import PdfReader


# ============================================
# Configuration
# ============================================

@dataclass
class Config:
    gemini_api_key: str
    docs_folder: str = "./docs"
    models_folder: str = "./models"
    cache_folder: str = "./cache"
    
    # Chunking settings
    chunk_size: int = 1000  # chars
    chunk_overlap: int = 100
    
    # Training settings
    num_queries: int = 150
    candidates_per_query: int = 20
    
    # Gemini settings
    embedding_model: str = "text-embedding-004"
    generation_model: str = "gemini-2.0-flash"
    
    # Rate limiting
    requests_per_minute: int = 30
    

# ============================================
# Data Classes
# ============================================

@dataclass
class Document:
    id: str
    filename: str
    content: str
    num_pages: int


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


@dataclass
class TrainingExample:
    query: str
    chunk_id: str
    features: Dict[str, float]
    label: int  # 1 = useful, 0 = not useful


# ============================================
# PDF Extraction
# ============================================

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, int]:
    """Extract text from PDF file."""
    try:
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts), len(reader.pages)
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return "", 0


def load_documents(docs_folder: str) -> List[Document]:
    """Load all PDFs from folder."""
    docs = []
    folder = Path(docs_folder)
    
    if not folder.exists():
        print(f"Folder {docs_folder} does not exist!")
        return docs
    
    pdf_files = list(folder.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")
    
    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        content, num_pages = extract_text_from_pdf(str(pdf_path))
        
        if content.strip():
            doc_id = hashlib.md5(str(pdf_path).encode()).hexdigest()[:12]
            docs.append(Document(
                id=doc_id,
                filename=pdf_path.name,
                content=content,
                num_pages=num_pages
            ))
        else:
            print(f"    Warning: No text extracted from {pdf_path.name}")
    
    print(f"Loaded {len(docs)} documents with content")
    return docs


# ============================================
# Chunking
# ============================================

def chunk_document(doc: Document, chunk_size: int = 1000, overlap: int = 100) -> List[Chunk]:
    """Split document into chunks."""
    text = doc.content
    chunks = []
    
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph exceeds chunk size, save current and start new
        if current_chunk and len(current_chunk) + len(para) > chunk_size:
            chunk_id = f"{doc.id}_{chunk_index}"
            chunks.append(Chunk(
                id=chunk_id,
                doc_id=doc.id,
                doc_filename=doc.filename,
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                total_chunks=0,  # Will update later
                token_estimate=len(current_chunk.split())
            ))
            chunk_index += 1
            
            # Start new chunk with overlap
            words = current_chunk.split()
            overlap_text = " ".join(words[-overlap//4:]) if len(words) > overlap//4 else ""
            current_chunk = overlap_text + " " + para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para
    
    # Don't forget last chunk
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
    
    # Update total_chunks
    for chunk in chunks:
        chunk.total_chunks = len(chunks)
    
    return chunks


# ============================================
# Gemini API Functions
# ============================================

class GeminiClient:
    def __init__(self, api_key: str, requests_per_minute: int = 30):
        self.api_key = api_key
        self.rpm = requests_per_minute
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start = time.time()
    
    def _rate_limit(self):
        """Simple rate limiting."""
        now = time.time()
        
        # Reset counter every minute
        if now - self.minute_start > 60:
            self.request_count = 0
            self.minute_start = now
        
        # If at limit, wait
        if self.request_count >= self.rpm:
            wait_time = 60 - (now - self.minute_start)
            if wait_time > 0:
                print(f"    Rate limit: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            self.request_count = 0
            self.minute_start = time.time()
        
        self.request_count += 1
    
    def embed(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text."""
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
                else:
                    print(f"    Embedding error: {response.status_code}")
                    return None
        except Exception as e:
            print(f"    Embedding exception: {e}")
            return None
    
    def generate(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate text with Gemini."""
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
                else:
                    print(f"    Generation error: {response.status_code} - {response.text[:200]}")
                    return ""
        except Exception as e:
            print(f"    Generation exception: {e}")
            return ""


# ============================================
# Query Generation
# ============================================

def generate_queries_for_doc(gemini: GeminiClient, doc: Document, num_queries: int = 5) -> List[str]:
    """Generate synthetic queries about a document."""
    
    # Use first ~3000 chars of doc for context
    doc_preview = doc.content[:3000]
    
    prompt = f"""Based on this document excerpt, generate {num_queries} diverse questions that someone might ask about this document's content.

Document: {doc.filename}
---
{doc_preview}
---

Generate {num_queries} questions. Make them varied:
- Some factual (who, what, when)
- Some conceptual (why, how, explain)
- Some specific details
- Some broad summaries

Output ONLY the questions, one per line, no numbering:"""

    response = gemini.generate(prompt, max_tokens=500)
    
    # Parse questions
    questions = [q.strip() for q in response.strip().split("\n") if q.strip() and "?" in q]
    return questions[:num_queries]


# ============================================
# Vector Similarity
# ============================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def retrieve_candidates(
    query_embedding: List[float],
    chunks: List[Chunk],
    top_k: int = 20
) -> List[Tuple[Chunk, float]]:
    """Retrieve top-k chunks by similarity."""
    
    scored = []
    for chunk in chunks:
        if chunk.embedding:
            sim = cosine_similarity(query_embedding, chunk.embedding)
            scored.append((chunk, sim))
    
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ============================================
# Feature Extraction
# ============================================

def extract_features(
    query: str,
    chunk: Chunk,
    similarity: float,
    all_chunks: List[Chunk]
) -> Dict[str, float]:
    """Extract features for a query-chunk pair."""
    
    # Normalize chunk position (0 = start, 1 = end)
    position = chunk.chunk_index / max(chunk.total_chunks, 1)
    
    # Keyword overlap
    query_words = set(query.lower().split())
    chunk_words = set(chunk.content.lower().split())
    common_words = query_words & chunk_words
    keyword_overlap = len(common_words) / max(len(query_words), 1)
    
    # Chunk length features
    chunk_length = len(chunk.content)
    token_count = chunk.token_estimate
    
    # Content features
    has_numbers = 1.0 if any(c.isdigit() for c in chunk.content) else 0.0
    has_bullet_points = 1.0 if any(x in chunk.content for x in ["•", "-", "*", "1.", "2."]) else 0.0
    sentence_count = chunk.content.count(".") + chunk.content.count("?") + chunk.content.count("!")
    avg_sentence_length = chunk_length / max(sentence_count, 1)
    
    # Query length
    query_length = len(query.split())
    
    return {
        "similarity": similarity,
        "position": position,
        "position_early": 1.0 if position < 0.3 else 0.0,
        "position_middle": 1.0 if 0.3 <= position < 0.7 else 0.0,
        "position_late": 1.0 if position >= 0.7 else 0.0,
        "keyword_overlap": keyword_overlap,
        "chunk_length": chunk_length / 2000,  # Normalize
        "token_count": token_count / 500,  # Normalize
        "has_numbers": has_numbers,
        "has_bullet_points": has_bullet_points,
        "sentence_count": sentence_count / 20,  # Normalize
        "avg_sentence_length": avg_sentence_length / 100,  # Normalize
        "query_length": query_length / 20,  # Normalize
        "length_ratio": chunk_length / (query_length * 50 + 1),  # Chunk length relative to query
    }


# ============================================
# LLM-as-Judge
# ============================================

def judge_chunk_utility(
    gemini: GeminiClient,
    query: str,
    answer: str,
    chunk: Chunk
) -> int:
    """Use LLM to judge if chunk was useful for answering query."""
    
    prompt = f"""You are evaluating if a text chunk was useful for answering a question.

Question: {query}

Answer that was generated: {answer[:1000]}

Chunk being evaluated:
---
{chunk.content[:1500]}
---

Was this chunk useful for generating the answer? Consider:
- Does the chunk contain information that appears in the answer?
- Does the chunk provide relevant context?
- Would removing this chunk make the answer worse?

Respond with ONLY "yes" or "no":"""

    response = gemini.generate(prompt, max_tokens=10).strip().lower()
    
    if "yes" in response:
        return 1
    else:
        return 0


def generate_answer(gemini: GeminiClient, query: str, chunks: List[Chunk]) -> str:
    """Generate answer using top chunks as context."""
    
    context = "\n\n---\n\n".join([c.content[:1000] for c in chunks[:5]])
    
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    return gemini.generate(prompt, max_tokens=500)


# ============================================
# Main Training Pipeline
# ============================================

def train_rf_model(
    config: Config,
    save_checkpoints: bool = True
):
    """Main training pipeline."""
    
    print("\n" + "="*60)
    print("RF Model Training for RAG Chunk Scoring")
    print("="*60)
    
    # Initialize Gemini client
    gemini = GeminiClient(config.gemini_api_key, config.requests_per_minute)
    
    # Create cache folder
    cache_folder = Path(config.cache_folder)
    cache_folder.mkdir(exist_ok=True)
    
    # =========================================
    # Step 1: Load Documents
    # =========================================
    print("\n[Step 1/6] Loading documents...")
    docs = load_documents(config.docs_folder)
    
    if not docs:
        print("No documents found! Add PDFs to the docs folder.")
        return
    
    # =========================================
    # Step 2: Chunk Documents
    # =========================================
    print("\n[Step 2/6] Chunking documents...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc, config.chunk_size, config.chunk_overlap)
        all_chunks.extend(chunks)
        print(f"  {doc.filename}: {len(chunks)} chunks")
    
    print(f"Total chunks: {len(all_chunks)}")
    
    # =========================================
    # Step 3: Generate Embeddings
    # =========================================
    print("\n[Step 3/6] Generating embeddings...")
    
    # Check for cached embeddings
    embeddings_cache_path = cache_folder / "embeddings_cache.pkl"
    if embeddings_cache_path.exists():
        print("  Loading cached embeddings...")
        with open(embeddings_cache_path, "rb") as f:
            embeddings_cache = pickle.load(f)
    else:
        embeddings_cache = {}
    
    chunks_to_embed = [c for c in all_chunks if c.id not in embeddings_cache]
    print(f"  Cached: {len(all_chunks) - len(chunks_to_embed)}, To embed: {len(chunks_to_embed)}")
    
    for i, chunk in enumerate(chunks_to_embed):
        if (i + 1) % 10 == 0:
            print(f"  Embedding chunk {i+1}/{len(chunks_to_embed)}...")
        
        embedding = gemini.embed(chunk.content)
        if embedding:
            embeddings_cache[chunk.id] = embedding
            chunk.embedding = embedding
    
    # Save cache
    with open(embeddings_cache_path, "wb") as f:
        pickle.dump(embeddings_cache, f)
    
    # Apply cached embeddings
    for chunk in all_chunks:
        if chunk.id in embeddings_cache:
            chunk.embedding = embeddings_cache[chunk.id]
    
    embedded_count = sum(1 for c in all_chunks if c.embedding)
    print(f"  Chunks with embeddings: {embedded_count}/{len(all_chunks)}")
    
    # =========================================
    # Step 4: Generate Queries
    # =========================================
    print("\n[Step 4/6] Generating synthetic queries...")
    
    queries_per_doc = max(1, config.num_queries // len(docs))
    all_queries = []
    
    for doc in docs:
        print(f"  Generating queries for: {doc.filename}")
        queries = generate_queries_for_doc(gemini, doc, queries_per_doc)
        for q in queries:
            all_queries.append({"query": q, "source_doc": doc.filename})
        print(f"    Generated {len(queries)} queries")
    
    print(f"Total queries: {len(all_queries)}")
    
    # =========================================
    # Step 5: Collect Training Data
    # =========================================
    print("\n[Step 5/6] Collecting training data (this takes a while)...")
    
    training_data = []
    
    for i, q_data in enumerate(all_queries):
        query = q_data["query"]
        
        if (i + 1) % 10 == 0:
            print(f"  Processing query {i+1}/{len(all_queries)}: {query[:50]}...")
        
        # Embed query
        query_embedding = gemini.embed(query)
        if not query_embedding:
            continue
        
        # Retrieve candidates
        candidates = retrieve_candidates(query_embedding, all_chunks, config.candidates_per_query)
        
        if len(candidates) < 5:
            continue
        
        # Generate answer using top 5
        top_chunks = [c for c, _ in candidates[:5]]
        answer = generate_answer(gemini, query, top_chunks)
        
        if not answer:
            continue
        
        # Judge each candidate chunk
        for chunk, similarity in candidates:
            # Extract features
            features = extract_features(query, chunk, similarity, all_chunks)
            
            # LLM judges utility
            label = judge_chunk_utility(gemini, query, answer, chunk)
            
            training_data.append(TrainingExample(
                query=query,
                chunk_id=chunk.id,
                features=features,
                label=label
            ))
        
        # Save checkpoint every 20 queries
        if save_checkpoints and (i + 1) % 20 == 0:
            checkpoint_path = cache_folder / f"training_data_checkpoint_{i+1}.pkl"
            with open(checkpoint_path, "wb") as f:
                pickle.dump(training_data, f)
            print(f"    Saved checkpoint: {len(training_data)} examples")
    
    print(f"\nTotal training examples: {len(training_data)}")
    
    # Save final training data
    training_data_path = cache_folder / "training_data_final.pkl"
    with open(training_data_path, "wb") as f:
        pickle.dump(training_data, f)
    
    # =========================================
    # Step 6: Train Random Forest
    # =========================================
    print("\n[Step 6/6] Training Random Forest model...")
    
    if len(training_data) < 50:
        print("Not enough training data! Need at least 50 examples.")
        return
    
    # Prepare data
    feature_names = list(training_data[0].features.keys())
    X = np.array([[ex.features[f] for f in feature_names] for ex in training_data])
    y = np.array([ex.label for ex in training_data])
    
    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {len(X)}")
    print(f"  Positive rate: {y.mean():.2%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n  Model Accuracy: {accuracy:.2%}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    print("\n  Feature Importance:")
    importances = sorted(zip(feature_names, rf.feature_importances_), key=lambda x: -x[1])
    for name, imp in importances:
        bar = "█" * int(imp * 50)
        print(f"    {name:25s} {imp:.3f} {bar}")
    
    # Save model
    models_folder = Path(config.models_folder)
    models_folder.mkdir(exist_ok=True)
    
    model_path = models_folder / "rf_model.pkl"
    model_data = {
        "model": rf,
        "feature_names": feature_names,
        "accuracy": accuracy,
        "training_samples": len(training_data),
        "positive_rate": float(y.mean())
    }
    joblib.dump(model_data, model_path)
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"   Accuracy: {accuracy:.2%}")
    print(f"   Training samples: {len(training_data)}")
    
    return model_data


# ============================================
# CLI Entry Point
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RF model for RAG chunk scoring")
    parser.add_argument("--api_key", type=str, help="Gemini API key")
    parser.add_argument("--docs_folder", type=str, default="./docs", help="Folder with PDF documents")
    parser.add_argument("--num_queries", type=int, default=150, help="Number of synthetic queries to generate")
    parser.add_argument("--rpm", type=int, default=30, help="Gemini requests per minute")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("VITE_GEMINI_API_KEY")
    
    if not api_key:
        print("Error: Please provide Gemini API key via --api_key or GEMINI_API_KEY env var")
        exit(1)
    
    config = Config(
        gemini_api_key=api_key,
        docs_folder=args.docs_folder,
        num_queries=args.num_queries,
        requests_per_minute=args.rpm
    )
    
    train_rf_model(config)

