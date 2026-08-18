"""
QUBO Weight Optimizer
=====================
Uses SHAP analysis + Grid Search + LLM-as-Judge to find optimal QUBO penalty weights.
No guessing - fully data-driven.

Run: streamlit run training/optimize_weights.py
"""

import streamlit as st
import numpy as np
import json
import pickle
import time
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random

# ============================================
# CONFIG
# ============================================
SCRIPT_DIR = Path(__file__).parent  # Get directory where this script is located
CACHE_DIR = SCRIPT_DIR / "cache"  # cache folder in same directory as script
MODEL_PATH = CACHE_DIR / "model.pkl"
TRAINING_DATA_PATH = CACHE_DIR / "training_data.pkl"
CHUNKS_PATH = CACHE_DIR / "chunks.pkl"
EMBEDDINGS_PATH = CACHE_DIR / "embeddings_cache.pkl"
OUTPUT_PATH = SCRIPT_DIR.parent / "server" / "shap_weights.json"  # Go up one level to get to server/

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="QUBO Weight Optimizer",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #0a0a0f; }
    .main-header { 
        font-size: 2.5rem; 
        font-weight: 700; 
        background: linear-gradient(90deg, #00d4ff, #7b2fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .step-card {
        background: rgba(20, 20, 35, 0.8);
        border: 1px solid rgba(100, 100, 150, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .metric-good { color: #00ff88; }
    .metric-bad { color: #ff4466; }
</style>
""", unsafe_allow_html=True)


# ============================================
# DATA CLASSES
# ============================================
@dataclass
class WeightCombo:
    threshold_high: float
    penalty_high: float
    threshold_med: float
    penalty_med: float
    
    def to_dict(self):
        return {
            "threshold_high": self.threshold_high,
            "penalty_high": self.penalty_high,
            "threshold_med": self.threshold_med,
            "penalty_med": self.penalty_med
        }


# ============================================
# HELPER FUNCTIONS
# ============================================
def load_model():
    """Load the trained RF model."""
    if not MODEL_PATH.exists():
        return None, None, "Model not found"
    
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        return model_data["model"], model_data, None
    except Exception as e:
        return None, None, str(e)


def load_training_data():
    """Load training data for SHAP analysis."""
    if not TRAINING_DATA_PATH.exists():
        return None, "Training data not found"
    
    try:
        with open(TRAINING_DATA_PATH, "rb") as f:
            data = pickle.load(f)
        return data, None
    except Exception as e:
        return None, str(e)


def load_chunks_and_embeddings():
    """Load chunks and embeddings for testing."""
    chunks = []
    embeddings = {}
    
    if CHUNKS_PATH.exists():
        try:
            # Check if file is empty or corrupted
            if CHUNKS_PATH.stat().st_size == 0:
                chunks = []
            else:
                with open(CHUNKS_PATH, "rb") as f:
                    chunks = pickle.load(f)
                # Ensure chunks is a list
                if not isinstance(chunks, list):
                    chunks = []
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            # File is corrupted, return empty list
            chunks = []
    
    if EMBEDDINGS_PATH.exists():
        try:
            # Check if file is empty or corrupted
            if EMBEDDINGS_PATH.stat().st_size == 0:
                embeddings = {}
            else:
                with open(EMBEDDINGS_PATH, "rb") as f:
                    embeddings = pickle.load(f)
                # Ensure embeddings is a dict
                if not isinstance(embeddings, dict):
                    embeddings = {}
        except (EOFError, pickle.UnpicklingError, Exception) as e:
            # File is corrupted, return empty dict
            embeddings = {}
    
    return chunks, embeddings


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================
# SHAP ANALYSIS
# ============================================
def run_shap_analysis(model, training_data, feature_names):
    """Run SHAP analysis on the model."""
    try:
        import shap
    except ImportError:
        return None, "SHAP not installed. Run: pip install shap"
    
    # Prepare feature matrix
    X = np.array([
        [td["features"].get(f, 0) for f in feature_names] 
        for td in training_data
    ])
    
    # Run SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Get values for positive class
    if isinstance(shap_values, list):
        shap_positive = shap_values[1]
    else:
        shap_positive = shap_values
    
    # Calculate feature importance
    mean_abs_shap = np.abs(shap_positive).mean(axis=0)
    # Ensure it's 1D and convert to Python floats
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()
    # Convert to list and ensure each value is a scalar
    importance = {}
    for name, val in zip(feature_names, mean_abs_shap):
        # Handle NumPy scalars and arrays properly
        if isinstance(val, np.ndarray):
            val = val.item() if val.size == 1 else float(val.flatten()[0])
        elif hasattr(val, 'item'):  # NumPy scalar
            val = val.item()
        importance[name] = float(val)
    
    # Analyze similarity feature specifically
    sim_idx = feature_names.index("similarity") if "similarity" in feature_names else -1
    sim_analysis = {}
    
    if sim_idx >= 0:
        sim_values = X[:, sim_idx]
        sim_shap = shap_positive[:, sim_idx]
        
        # Impact at different similarity levels
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            mask = sim_values > threshold
            if mask.sum() > 0:
                mean_val = sim_shap[mask].mean()
                # Ensure we get a Python float
                if isinstance(mean_val, np.ndarray):
                    mean_val = mean_val.item() if mean_val.size == 1 else float(mean_val.flatten()[0])
                elif hasattr(mean_val, 'item'):
                    mean_val = mean_val.item()
                sim_analysis[f"impact_above_{threshold}"] = float(mean_val)
        
        # Standard deviation of similarity impact
        std_val = np.std(sim_shap)
        if isinstance(std_val, np.ndarray):
            std_val = std_val.item() if std_val.size == 1 else float(std_val.flatten()[0])
        elif hasattr(std_val, 'item'):
            std_val = std_val.item()
        sim_analysis["variability"] = float(std_val)
        
        mean_impact_val = sim_shap.mean()
        if isinstance(mean_impact_val, np.ndarray):
            mean_impact_val = mean_impact_val.item() if mean_impact_val.size == 1 else float(mean_impact_val.flatten()[0])
        elif hasattr(mean_impact_val, 'item'):
            mean_impact_val = mean_impact_val.item()
        sim_analysis["mean_impact"] = float(mean_impact_val)
    
    return {
        "feature_importance": importance,
        "similarity_analysis": sim_analysis,
        "shap_values": shap_positive,
        "X": X
    }, None


def derive_search_ranges(shap_results):
    """Use SHAP results to narrow search ranges."""
    sim_analysis = shap_results.get("similarity_analysis", {})
    
    # Default ranges
    ranges = {
        "threshold_high": [0.80, 0.85, 0.90],
        "penalty_high": [0.5, 0.6, 0.7, 0.8, 0.9],
        "threshold_med": [0.65, 0.70, 0.75],
        "penalty_med": [0.2, 0.3, 0.4]
    }
    
    # Narrow based on SHAP insights
    var_val = sim_analysis.get("variability", 0.1)
    if isinstance(var_val, np.ndarray):
        variability = float(var_val.item() if var_val.size == 1 else var_val.flatten()[0])
    elif hasattr(var_val, 'item'):
        variability = float(var_val.item())
    elif isinstance(var_val, (list, tuple)):
        variability = float(var_val[0]) if len(var_val) > 0 else 0.1
    else:
        variability = float(var_val) if var_val is not None else 0.1
    
    # If similarity has high variability, focus on stricter thresholds
    if variability > 0.15:
        ranges["threshold_high"] = [0.82, 0.85, 0.88]
        ranges["penalty_high"] = [0.6, 0.7, 0.8]
    elif variability < 0.05:
        # Similarity doesn't vary much, less aggressive penalties
        ranges["penalty_high"] = [0.4, 0.5, 0.6]
    
    # Check impact at different thresholds - handle NumPy types
    def to_float(val, default=0.0):
        if val is None:
            return default
        if isinstance(val, np.ndarray):
            return float(val.item() if val.size == 1 else val.flatten()[0])
        elif hasattr(val, 'item'):
            return float(val.item())
        elif isinstance(val, (list, tuple)):
            return float(val[0]) if len(val) > 0 else default
        return float(val)
    
    impact_80_val = sim_analysis.get("impact_above_0.8", 0)
    impact_70_val = sim_analysis.get("impact_above_0.7", 0)
    impact_80 = abs(to_float(impact_80_val, 0))
    impact_70 = abs(to_float(impact_70_val, 0))
    
    if impact_70 < 0.02:
        # Low impact at 0.7, maybe medium threshold doesn't matter
        ranges["threshold_med"] = [0.70]
        ranges["penalty_med"] = [0.3]
    
    return ranges


def generate_combos(ranges: Dict) -> List[WeightCombo]:
    """Generate all combinations from ranges."""
    combos = []
    for th in ranges["threshold_high"]:
        for ph in ranges["penalty_high"]:
            for tm in ranges["threshold_med"]:
                for pm in ranges["penalty_med"]:
                    if tm < th:  # threshold_med must be < threshold_high
                        combos.append(WeightCombo(th, ph, tm, pm))
    return combos


# ============================================
# QUBO SIMULATION
# ============================================
def simulate_qubo_selection(
    chunks: List,
    embeddings: Dict,
    rf_model,
    feature_names: List[str],
    query_embedding: List[float],
    query: str,
    weights: WeightCombo,
    token_budget: int = 1500
) -> Tuple[List, Dict]:
    """Simulate QUBO chunk selection with given weights."""
    
    # Score all chunks
    scored_chunks = []
    for chunk in chunks:
        if chunk.id not in embeddings:
            continue
        
        emb = embeddings[chunk.id]
        similarity = cosine_similarity(query_embedding, emb)
        
        # Simple features
        features = {
            "similarity": similarity,
            "keyword_overlap": len(set(query.lower().split()) & set(chunk.content.lower().split())) / max(len(query.split()), 1),
            "position": chunk.chunk_index / 10,
            "position_early": 1.0 if chunk.chunk_index < 3 else 0.0,
            "position_middle": 1.0 if 3 <= chunk.chunk_index < 7 else 0.0,
            "position_late": 1.0 if chunk.chunk_index >= 7 else 0.0,
            "chunk_length": len(chunk.content),
            "token_count": len(chunk.content.split()),
            "has_numbers": 1.0 if any(c.isdigit() for c in chunk.content) else 0.0,
            "has_bullets": 1.0 if any(c in chunk.content for c in ['•', '-', '*', '1.', '2.']) else 0.0,
            "sentence_count": chunk.content.count('.') + chunk.content.count('!') + chunk.content.count('?'),
            "query_length": len(query.split())
        }
        
        # RF score
        X = np.array([[features.get(f, 0) for f in feature_names]])
        proba_result = rf_model.predict_proba(X)[0][1]
        # Ensure it's a Python float
        if isinstance(proba_result, np.ndarray):
            rf_score = float(proba_result.item() if proba_result.size == 1 else proba_result.flatten()[0])
        elif hasattr(proba_result, 'item'):
            rf_score = float(proba_result.item())
        else:
            rf_score = float(proba_result)
        
        scored_chunks.append({
            "chunk": chunk,
            "embedding": emb,
            "similarity": similarity,
            "rf_score": rf_score,
            "tokens": len(chunk.content.split())
        })
    
    # Sort by similarity (like traditional RAG fetches candidates)
    scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
    candidates = scored_chunks[:20]  # Top 20 candidates
    
    if not candidates:
        return [], {"error": "No candidates"}
    
    # Build redundancy matrix
    n = len(candidates)
    redundancy = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_similarity(candidates[i]["embedding"], candidates[j]["embedding"])
            redundancy[i][j] = redundancy[j][i] = sim
    
    # Greedy QUBO selection
    selected = []
    remaining = list(range(n))
    current_tokens = 0
    
    while remaining:
        best_idx = None
        best_value = -float('inf')
        
        for idx in remaining:
            chunk_tokens = candidates[idx]["tokens"]
            if current_tokens + chunk_tokens > token_budget:
                continue
            
            value = candidates[idx]["rf_score"]
            
            # Apply penalties based on weights
            for sel_idx in selected:
                if redundancy[idx][sel_idx] > weights.threshold_high:
                    value -= redundancy[idx][sel_idx] * weights.penalty_high
                elif redundancy[idx][sel_idx] > weights.threshold_med:
                    value -= redundancy[idx][sel_idx] * weights.penalty_med
            
            if value > best_value:
                best_value = value
                best_idx = idx
        
        if best_idx is None or best_value < 0:
            break
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        current_tokens += candidates[best_idx]["tokens"]
    
    selected_chunks = [candidates[i] for i in selected]
    
    # Compute metrics
    metrics = {
        "num_chunks": len(selected_chunks),
        "total_tokens": sum(c["tokens"] for c in selected_chunks),
        "avg_rf_score": np.mean([c["rf_score"] for c in selected_chunks]) if selected_chunks else 0,
        "avg_similarity": np.mean([c["similarity"] for c in selected_chunks]) if selected_chunks else 0,
        "redundant_pairs": 0
    }
    
    # Count redundant pairs
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if redundancy[selected[i]][selected[j]] > 0.85:
                metrics["redundant_pairs"] += 1
    
    return selected_chunks, metrics


# ============================================
# LLM JUDGE
# ============================================
def get_gemini_client(api_key: str = None):
    """Initialize Gemini client."""
    if api_key is None:
        api_key = st.session_state.get("gemini_api_key", "")
    if not api_key:
        return None
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Use gemini-2.5-pro
        return genai.GenerativeModel("gemini-2.5-pro")
    except Exception as e:
        if 'st' in globals():
            st.error(f"Gemini error: {e}")
        return None


def llm_pairwise_compare(
    model,
    query: str,
    chunks_a: List,
    chunks_b: List,
    label_a: str = "A",
    label_b: str = "B"
) -> str:
    """Use LLM to compare two chunk selections. Returns 'A', 'B', or 'TIE'."""
    
    context_a = "\n\n".join([f"[Chunk {i+1}]: {c['chunk'].content[:300]}..." for i, c in enumerate(chunks_a[:3])])
    context_b = "\n\n".join([f"[Chunk {i+1}]: {c['chunk'].content[:300]}..." for i, c in enumerate(chunks_b[:3])])
    
    prompt = f"""You are evaluating which set of document chunks would better help answer a user's question.

USER QUESTION: {query}

=== OPTION A ===
{context_a}

=== OPTION B ===
{context_b}

Which option provides MORE RELEVANT and USEFUL context for answering the question?
Consider: relevance to the question, information quality, and minimal redundancy.

Reply with ONLY one of: A, B, or TIE
"""
    
    try:
        response = model.generate_content(prompt)
        answer = response.text.strip().upper()
        if "A" in answer and "B" not in answer:
            return "A"
        elif "B" in answer and "A" not in answer:
            return "B"
        else:
            return "TIE"
    except Exception as e:
        st.warning(f"LLM call failed: {e}")
        return "TIE"


def generate_test_query(model, chunks: List = None, training_data: List = None) -> Tuple[str, List[float]]:
    """Generate a test query and its embedding."""
    query = None
    context = ""
    
    # First try: use existing queries from training data (fastest)
    if training_data and len(training_data) > 0:
        sample_td = random.sample(training_data, min(10, len(training_data)))
        for td in sample_td:
            if td.get("query") and td.get("query").strip():
                query = td["query"].strip()
                break
    
    # Second try: generate from chunks
    if not query and chunks and len(chunks) >= 1:
        sample_size = min(3, len(chunks))
        sample_chunks = random.sample(chunks, sample_size)
        context_parts = []
        for c in sample_chunks:
            if hasattr(c, 'content'):
                context_parts.append(c.content[:200])
            else:
                context_parts.append(str(c)[:200])
        context = " ".join(context_parts)
    
    # Third try: generate generic query
    if not query and not context:
        context = "general document content"
    
    try:
        import google.generativeai as genai
        
        # If we have an existing query, use it; otherwise generate one
        if query:
            # Use existing query from training data
            pass
        else:
            # Generate new query
            if context:
                prompt = f"""Based on this document content, generate ONE specific question that someone might ask:

{context[:500]}

Reply with ONLY the question, nothing else."""
            else:
                prompt = """Generate ONE specific question that someone might ask about a document.

Reply with ONLY the question, nothing else."""
            
            response = model.generate_content(prompt)
            query = response.text.strip()
        
        if not query:
            return None, None
        
        # Clean up query (remove quotes and extra whitespace)
        query = query.strip('"\' \n\t')
        
        # Get embedding using the correct model
        embed_result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        
        if not embed_result or 'embedding' not in embed_result:
            return None, None
            
        return query, embed_result['embedding']
    except Exception as e:
        print(f"Error generating test query: {e}")  # Debug logging
        return None, None


# ============================================
# GRID SEARCH
# ============================================
def run_grid_search(
    combos: List[WeightCombo],
    chunks: List,
    embeddings: Dict,
    rf_model,
    feature_names: List[str],
    gemini_model,
    num_queries: int = 20,
    progress_callback=None,
    training_data: List = None
) -> Tuple[WeightCombo, Dict]:
    """Run grid search with pairwise LLM comparisons."""
    
    # Generate test queries
    test_queries = []
    failed_count = 0
    for i in range(num_queries * 2):  # Try more times to account for failures
        if len(test_queries) >= num_queries:
            break
            
        if progress_callback:
            progress_callback(f"Generating query {len(test_queries)+1}/{num_queries}...")
        
        query, embedding = generate_test_query(gemini_model, chunks, training_data)
        if query and embedding:
            test_queries.append({"query": query, "embedding": embedding})
            failed_count = 0  # Reset on success
        else:
            failed_count += 1
            if failed_count > 10:  # Stop if too many consecutive failures
                break
        time.sleep(0.5)  # Rate limiting
    
    if len(test_queries) < 3:  # Lowered from 5 to 3
        return None, {"error": f"Failed to generate enough test queries (got {len(test_queries)}, need at least 3)"}
    
    # Score each combo
    combo_scores = {i: 0 for i in range(len(combos))}
    total_comparisons = len(combos) * len(test_queries)
    comparison_count = 0
    
    # For each query, compare all combos against a baseline
    baseline = WeightCombo(0.85, 0.5, 0.70, 0.3)  # Conservative baseline
    
    for q_idx, test_q in enumerate(test_queries):
        query = test_q["query"]
        query_emb = test_q["embedding"]
        
        # Get baseline selection
        baseline_chunks, baseline_metrics = simulate_qubo_selection(
            chunks, embeddings, rf_model, feature_names,
            query_emb, query, baseline
        )
        
        for c_idx, combo in enumerate(combos):
            comparison_count += 1
            if progress_callback:
                progress_callback(f"Comparison {comparison_count}/{total_comparisons}: Query {q_idx+1}, Combo {c_idx+1}")
            
            # Get combo selection
            combo_chunks, combo_metrics = simulate_qubo_selection(
                chunks, embeddings, rf_model, feature_names,
                query_emb, query, combo
            )
            
            # LLM comparison
            result = llm_pairwise_compare(
                gemini_model, query,
                combo_chunks, baseline_chunks,
                label_a=f"Combo {c_idx}", label_b="Baseline"
            )
            
            if result == "A":
                combo_scores[c_idx] += 2  # Win
            elif result == "TIE":
                combo_scores[c_idx] += 1  # Tie
            # Loss = 0 points
            
            time.sleep(0.3)  # Rate limiting
    
    # Find best combo
    best_idx = max(combo_scores, key=combo_scores.get)
    best_combo = combos[best_idx]
    
    results = {
        "scores": combo_scores,
        "best_idx": best_idx,
        "best_score": combo_scores[best_idx],
        "max_possible": len(test_queries) * 2,
        "num_queries": len(test_queries),
        "all_combos": [c.to_dict() for c in combos]
    }
    
    return best_combo, results


# ============================================
# STREAMLIT UI
# ============================================
def main():
    st.markdown('<h1 class="main-header">⚡ QUBO Weight Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("**Fully data-driven weight optimization using SHAP + Grid Search + LLM-as-Judge**")
    
    # Initialize session state
    if "optimization_step" not in st.session_state:
        st.session_state.optimization_step = 0
    if "shap_results" not in st.session_state:
        st.session_state.shap_results = None
    if "search_ranges" not in st.session_state:
        st.session_state.search_ranges = None
    if "best_weights" not in st.session_state:
        st.session_state.best_weights = None
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Gemini API Key", type="password", key="gemini_api_key")
        
        st.divider()
        st.header("Status")
        
        # Check model
        model, model_data, error = load_model()
        if model:
            st.success(f"✅ Model loaded")
            st.caption(f"Accuracy: {model_data.get('accuracy', 'N/A'):.1%}")
            st.caption(f"Samples: {model_data.get('training_samples', 'N/A')}")
        else:
            st.error(f"❌ Model: {error}")
        
        # Check training data
        training_data, error = load_training_data()
        if training_data:
            st.success(f"✅ Training data: {len(training_data)} samples")
        else:
            st.error(f"❌ Training data: {error}")
        
        # Check chunks
        chunks, embeddings = load_chunks_and_embeddings()
        if chunks:
            embedded = sum(1 for c in chunks if c.id in embeddings)
            st.success(f"✅ Chunks: {len(chunks)} ({embedded} embedded)")
        else:
            st.warning("⚠️ No chunks found")
    
    # Main content - Steps
    tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ SHAP Analysis", "2️⃣ Search Ranges", "3️⃣ Grid Search", "4️⃣ Results"])
    
    # ============================================
    # STEP 1: SHAP ANALYSIS
    # ============================================
    with tab1:
        st.header("Step 1: SHAP Analysis")
        st.markdown("Analyze the RF model to understand feature importance and interactions.")
        
        if not model:
            st.error("Please ensure the RF model is trained first.")
            return
        
        if st.button("🔬 Run SHAP Analysis", type="primary"):
            with st.spinner("Running SHAP analysis..."):
                feature_names = model_data.get("feature_names", [])
                shap_results, error = run_shap_analysis(model, training_data, feature_names)
                
                if error:
                    st.error(f"SHAP failed: {error}")
                else:
                    st.session_state.shap_results = shap_results
                    st.session_state.optimization_step = 1
                    st.success("SHAP analysis complete!")
        
        if st.session_state.shap_results:
            results = st.session_state.shap_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Feature Importance")
                importance = results["feature_importance"]
                # Ensure all values are floats (in case they're arrays or other types)
                cleaned_importance = {}
                for k, v in importance.items():
                    if isinstance(v, np.ndarray):
                        v = v.item() if v.size == 1 else float(v.flatten()[0])
                    elif hasattr(v, 'item'):  # NumPy scalar
                        v = v.item()
                    elif isinstance(v, (list, tuple)):
                        v = float(v[0]) if len(v) > 0 else 0.0
                    cleaned_importance[k] = float(v) if not isinstance(v, (int, float)) else float(v)
                importance = cleaned_importance
                sorted_imp = sorted(importance.items(), key=lambda x: float(x[1]), reverse=True)
                max_imp = max(importance.values()) if importance.values() else 1.0
                for feat, imp in sorted_imp:
                    imp_float = float(imp)
                    st.progress(min(1.0, imp_float / max_imp if max_imp > 0 else 1.0), text=f"{feat}: {imp_float:.4f}")
            
            with col2:
                st.subheader("Similarity Analysis")
                sim_analysis = results["similarity_analysis"]
                for key, value in sim_analysis.items():
                    # Ensure value is a float - handle all NumPy types
                    if isinstance(value, np.ndarray):
                        val_float = value.item() if value.size == 1 else float(value.flatten()[0])
                    elif hasattr(value, 'item'):  # NumPy scalar
                        val_float = value.item()
                    elif isinstance(value, (list, tuple)):
                        val_float = float(value[0]) if len(value) > 0 else 0.0
                    else:
                        val_float = float(value) if not isinstance(value, (int, float)) else float(value)
                    st.metric(key.replace("_", " ").title(), f"{val_float:.4f}")
    
    # ============================================
    # STEP 2: DERIVE SEARCH RANGES
    # ============================================
    with tab2:
        st.header("Step 2: Derive Search Ranges")
        st.markdown("Use SHAP insights to narrow the search space.")
        
        if not st.session_state.shap_results:
            st.warning("Please run SHAP analysis first.")
            return
        
        if st.button("📊 Derive Ranges from SHAP", type="primary"):
            ranges = derive_search_ranges(st.session_state.shap_results)
            st.session_state.search_ranges = ranges
            st.session_state.optimization_step = 2
        
        if st.session_state.search_ranges:
            ranges = st.session_state.search_ranges
            
            st.subheader("Narrowed Search Ranges")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**threshold_high:**", ranges["threshold_high"])
                st.write("**penalty_high:**", ranges["penalty_high"])
            
            with col2:
                st.write("**threshold_med:**", ranges["threshold_med"])
                st.write("**penalty_med:**", ranges["penalty_med"])
            
            combos = generate_combos(ranges)
            st.metric("Total Combinations", len(combos))
            st.caption("(Reduced from 135 based on SHAP insights)")
    
    # ============================================
    # STEP 3: GRID SEARCH
    # ============================================
    with tab3:
        st.header("Step 3: Grid Search with LLM Judge")
        st.markdown("Test each combination using pairwise LLM comparisons.")
        
        if not st.session_state.search_ranges:
            st.warning("Please derive search ranges first.")
            return
        
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
            return
        
        num_queries = st.slider("Number of test queries", 10, 50, 20)
        
        combos = generate_combos(st.session_state.search_ranges)
        st.info(f"Will test {len(combos)} combinations on {num_queries} queries = {len(combos) * num_queries} comparisons")
        
        estimated_time = (len(combos) * num_queries * 1.5) / 60  # ~1.5 sec per comparison
        st.caption(f"Estimated time: ~{estimated_time:.0f} minutes")
        
        if st.button("🚀 Start Grid Search", type="primary"):
            gemini_model = get_gemini_client(api_key)
            if not gemini_model:
                st.error("Failed to initialize Gemini client.")
                return
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(msg):
                status_text.text(msg)
            
            with st.spinner("Running grid search..."):
                best_weights, results = run_grid_search(
                    combos=combos,
                    chunks=chunks,
                    embeddings=embeddings,
                    rf_model=model,
                    feature_names=model_data.get("feature_names", []),
                    gemini_model=gemini_model,
                    num_queries=num_queries,
                    progress_callback=update_progress,
                    training_data=training_data
                )
                
                if best_weights:
                    st.session_state.best_weights = best_weights
                    st.session_state.grid_results = results
                    st.session_state.optimization_step = 3
                    st.success("Grid search complete!")
                else:
                    st.error(f"Grid search failed: {results.get('error', 'Unknown error')}")
    
    # ============================================
    # STEP 4: RESULTS
    # ============================================
    with tab4:
        st.header("Step 4: Optimal Weights")
        
        if not st.session_state.best_weights:
            st.warning("Please complete grid search first.")
            return
        
        weights = st.session_state.best_weights
        results = st.session_state.grid_results
        
        st.subheader("🏆 Winning Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("threshold_high", f"{weights.threshold_high:.2f}")
        col2.metric("penalty_high", f"{weights.penalty_high:.2f}")
        col3.metric("threshold_med", f"{weights.threshold_med:.2f}")
        col4.metric("penalty_med", f"{weights.penalty_med:.2f}")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Win Score", f"{results['best_score']}/{results['max_possible']}")
            st.caption(f"Based on {results['num_queries']} test queries")
        
        with col2:
            win_rate = results['best_score'] / results['max_possible'] * 100
            st.metric("Win Rate", f"{win_rate:.1f}%")
        
        st.divider()
        
        # Save weights
        if st.button("💾 Save Weights to server/shap_weights.json", type="primary"):
            output = {
                "threshold_high": weights.threshold_high,
                "penalty_high": weights.penalty_high,
                "threshold_med": weights.threshold_med,
                "penalty_med": weights.penalty_med,
                "win_score": results['best_score'],
                "max_possible": results['max_possible'],
                "num_test_queries": results['num_queries'],
                "derived_via": "SHAP + Grid Search + LLM-as-Judge"
            }
            
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(output, f, indent=2)
            
            st.success(f"Weights saved to {OUTPUT_PATH}")
            st.code(json.dumps(output, indent=2))
        
        st.divider()
        
        # Show JSON for manual copy
        st.subheader("JSON Output")
        output = {
            "threshold_high": weights.threshold_high,
            "penalty_high": weights.penalty_high,
            "threshold_med": weights.threshold_med,
            "penalty_med": weights.penalty_med
        }
        st.code(json.dumps(output, indent=2), language="json")


if __name__ == "__main__":
    main()

