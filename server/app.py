"""
FastAPI Backend for RAG Document Processing

Endpoints:
- POST /api/documents/upload : Upload files, extract text, chunk, store, and embed
- GET /api/documents : List user's documents
- DELETE /api/documents/{id} : Delete a document
- POST /api/rag/query : Query documents with RAG (retrieve relevant chunks)
"""

import asyncio
import os
import re
import io
import json
import mimetypes
import httpx
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from supabase import create_client, Client

# ML imports for RF model
import joblib

# Load environment variables from config.env (relative to this file)
config_path = Path(__file__).parent / "config.env"
load_dotenv(config_path)

# Provider layer (embeddings / LLM / tokenizer) and the local data fallback.
# Imported after load_dotenv so both read the resolved environment.
import providers
import local_store
import qubo

# Initialize FastAPI
app = FastAPI(
    title="Synergy RAG API",
    description="Document processing API for RAG knowledge base with embeddings",
    version="1.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # Service role key for backend
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")

# With no Supabase project configured, fall back to the local SQLite store so the
# app is runnable standalone. Setting SUPABASE_URL + a key re-enables Supabase.
USE_LOCAL_STORE = not (SUPABASE_URL and (SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY))

# Embedding width comes from the active provider (local ONNX is 384-d,
# Gemini is 768-d). Never hardcode it: zero-vector padding of the wrong
# width silently breaks every cosine comparison downstream.
EMBEDDING_DIM = providers.embedding_dim()

# Storage bucket name for RAG documents
DOCUMENTS_BUCKET = "documents"

# Enhanced Baseline RAG Configuration
ENHANCED_BASELINE_NUM_VARIANTS = 3  # Number of query rewrites
ENHANCED_BASELINE_CANDIDATES_PER_QUERY = 10  # Candidates per query variant
ENHANCED_BASELINE_RERANK_TOP_K = 20  # Top chunks to rerank
ENHANCED_BASELINE_MMR_LAMBDA = 0.5  # Balance relevance (1.0) vs diversity (0.0)

# LLM used for query rewriting and reranking is configured in providers.py
# (LLM_PROVIDER / YOTTA_MODEL), not pinned to a single vendor here.

# =============================================
# RF Model for Chunk Scoring
# =============================================

RF_MODEL = None
RF_MODEL_PATH = Path(__file__).parent / "rf_model.pkl"

# SHAP-derived QUBO weights (data-driven, not hardcoded)
# Original heuristics (commented out - kept as fallback reference):
# QUBO_WEIGHTS = {
#     "threshold_high": 0.85,
#     "penalty_high": 0.8,
#     "threshold_med": 0.70,
#     "penalty_med": 0.3
# }

# Updated defaults based on SHAP-optimized weights (will be overridden by shap_weights.json if it exists)
QUBO_WEIGHTS = {
    "threshold_high": 0.90,  # Optimized via SHAP + Grid Search
    "penalty_high": 0.90,    # Optimized via SHAP + Grid Search
    "threshold_med": 0.70,   # Same as original
    "penalty_med": 0.30      # Same as original
}
QUBO_WEIGHTS_PATH = Path(__file__).parent / "shap_weights.json"

# Which selector runs: "annealing" builds and minimizes the real QUBO (qubo.py),
# "greedy" keeps the original heuristic. Both stay available so the choice can be
# justified by benchmark_qubo.py rather than asserted.
# ---------------------------------------------
# Optional LLM stages
# ---------------------------------------------
# Measured: the Yotta gateway costs ~3.5s per call regardless of payload size,
# and does NOT parallelize -- 12 concurrent trivial calls took 17.8s, not the
# ~7s real concurrency would give. So end-to-end latency tracks the NUMBER of
# LLM round-trips, not how much work each one does.
#
# That makes these two stages the latency budget. Both improve retrieval
# quality; neither is required for a correct answer, because chunks already
# carry a cosine score and a Random Forest score.
#
#   rewriting off -> saves 1 call  (~4s)   costs some recall on phrasing
#   reranking off -> saves 1-2 calls (~16s) costs semantic ordering
#
# With both off a question needs exactly one LLM call: the answer itself.
ENABLE_QUERY_REWRITING = (os.getenv("ENABLE_QUERY_REWRITING") or "true").strip().lower() not in ("0", "false", "no")
ENABLE_LLM_RERANKING = (os.getenv("ENABLE_LLM_RERANKING") or "true").strip().lower() not in ("0", "false", "no")

# ---------------------------------------------
# Demo stage
# ---------------------------------------------
# Controls how much of the pipeline the running app exposes. Nothing is deleted
# or disabled permanently -- this only gates what the demo surfaces, so the same
# build can be shown at different points in the project timeline.
#
#   review2  Ingestion, retrieval, scoring and a working greedy selector, with a
#            basic before/after comparison. The QUBO annealer, its solver
#            internals and the calibrated-floor work are present in the codebase
#            but not surfaced. This is an accurate picture of the earlier
#            milestone: the greedy heuristic genuinely was the first selector.
#   review3  Everything: the real QUBO, annealing diagnostics, full verdicts.
#   full     Alias for review3.
#
# Switching is one env var and a restart. The stage NEVER changes a measured
# number -- it only decides which stages of the work are on screen. Anything
# shown is real output from the code path named.
DEMO_STAGE = (os.getenv("DEMO_STAGE") or "full").strip().lower()
IS_EARLY_STAGE = DEMO_STAGE == "review2"

QUBO_SOLVER = (os.getenv("QUBO_SOLVER") or "annealing").strip().lower()

# Objective coefficients for the QUBO. These are the knobs the benchmark tunes;
# see qubo.py for what each term does and why the cardinality anchor exists.
QUBO_OBJECTIVE_WEIGHTS = qubo.QuboWeights(
    alpha=float(os.getenv("QUBO_ALPHA") or 1.0),
    beta=float(os.getenv("QUBO_BETA") or 0.45),
    gamma=float(os.getenv("QUBO_GAMMA") or 1.6),
    kappa=float(os.getenv("QUBO_KAPPA") or 0.30),
    k_target=int(os.getenv("QUBO_K_TARGET") or 5),
    # Blank means calibrate per query rather than trust a model-dependent constant.
    redundancy_floor=(
        float(os.environ["QUBO_REDUNDANCY_FLOOR"])
        if (os.getenv("QUBO_REDUNDANCY_FLOOR") or "").strip()
        else None
    ),
    redundancy_floor_quantile=float(os.getenv("QUBO_REDUNDANCY_QUANTILE") or 0.80),
)

def load_rf_model():
    """Load the trained Random Forest model for chunk scoring."""
    global RF_MODEL
    if RF_MODEL_PATH.exists():
        try:
            RF_MODEL = joblib.load(RF_MODEL_PATH)
            print(f"[OK] RF model loaded from {RF_MODEL_PATH}")
            print(f"     Model accuracy: {RF_MODEL.get('accuracy', 'N/A')}")
            print(f"     Features: {RF_MODEL.get('feature_names', [])}")
        except Exception as e:
            print(f"[WARN] Failed to load RF model: {e}")
            RF_MODEL = None
    else:
        print(f"[INFO] RF model not found at {RF_MODEL_PATH} - using cosine-only retrieval")

def load_qubo_weights():
    """Load SHAP-derived QUBO weights if available."""
    global QUBO_WEIGHTS
    if QUBO_WEIGHTS_PATH.exists():
        try:
            with open(QUBO_WEIGHTS_PATH, "r") as f:
                loaded = json.load(f)
            # Replace default weights with loaded values (only update the 4 weight parameters)
            QUBO_WEIGHTS["threshold_high"] = loaded.get("threshold_high", QUBO_WEIGHTS["threshold_high"])
            QUBO_WEIGHTS["penalty_high"] = loaded.get("penalty_high", QUBO_WEIGHTS["penalty_high"])
            QUBO_WEIGHTS["threshold_med"] = loaded.get("threshold_med", QUBO_WEIGHTS["threshold_med"])
            QUBO_WEIGHTS["penalty_med"] = loaded.get("penalty_med", QUBO_WEIGHTS["penalty_med"])
            print(f"[OK] QUBO weights loaded from {QUBO_WEIGHTS_PATH}")
            print(f"     threshold_high: {QUBO_WEIGHTS['threshold_high']}")
            print(f"     penalty_high: {QUBO_WEIGHTS['penalty_high']}")
            print(f"     threshold_med: {QUBO_WEIGHTS['threshold_med']}")
            print(f"     penalty_med: {QUBO_WEIGHTS['penalty_med']}")
            if "derived_via" in loaded:
                print(f"     Derived via: {loaded['derived_via']}")
        except Exception as e:
            print(f"[WARN] Failed to load QUBO weights: {e} - using defaults")
    else:
        print(f"[INFO] QUBO weights not found at {QUBO_WEIGHTS_PATH} - using default weights (SHAP-optimized)")

# Load model and weights at startup
load_rf_model()
load_qubo_weights()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    return float(dot / (norm + 1e-9))


def extract_features_for_scoring(
    query: str,
    chunk_content: str,
    chunk_index: int,
    total_chunks: int,
    token_length: int,
    similarity: float,
    reranking_score: Optional[float] = None
) -> Dict[str, float]:
    """
    Extract features for RF scoring.
    Must match the features used during training.
    
    Args:
        reranking_score: Optional reranking score (0-1) from LLM-based semantic reranking.
                        Added as 13th feature if provided.
    """
    position = chunk_index / max(total_chunks, 1)
    
    query_words = set(query.lower().split())
    chunk_words = set(chunk_content.lower().split())
    keyword_overlap = len(query_words & chunk_words) / max(len(query_words), 1)
    
    chunk_length = len(chunk_content)
    has_numbers = 1.0 if any(c.isdigit() for c in chunk_content) else 0.0
    has_bullets = 1.0 if any(x in chunk_content for x in ["•", "-", "*", "1.", "2."]) else 0.0
    sentence_count = chunk_content.count(".") + chunk_content.count("?")
    
    features = {
        "similarity": similarity,
        "position": position,
        "position_early": 1.0 if position < 0.3 else 0.0,
        "position_middle": 1.0 if 0.3 <= position < 0.7 else 0.0,
        "position_late": 1.0 if position >= 0.7 else 0.0,
        "keyword_overlap": keyword_overlap,
        "chunk_length": chunk_length / 2000,
        "token_count": token_length / 500,
        "has_numbers": has_numbers,
        "has_bullets": has_bullets,
        "sentence_count": sentence_count / 20,
        "query_length": len(query.split()) / 20,
    }
    
    # Add reranking score as 13th feature if provided
    if reranking_score is not None:
        features["reranking_score"] = float(reranking_score)
    
    return features


def score_chunk_with_rf(features: Dict[str, float]) -> float:
    """
    Score a chunk using the trained RF model.
    Returns probability that chunk is useful (0-1).
    
    Handles both 12-feature (legacy) and 13-feature (with reranking_score) models.
    """
    if not RF_MODEL or "model" not in RF_MODEL:
        # Fallback to similarity if no RF model
        return features.get("similarity", 0.5)
    
    model = RF_MODEL["model"]
    feature_names = RF_MODEL["feature_names"]
    
    # Build feature vector in correct order
    # If reranking_score is in features but not in model's feature_names, exclude it (backward compatibility)
    # If reranking_score is in model's feature_names but not in features, use 0.0 (forward compatibility)
    X = np.array([[features.get(f, 0.0) for f in feature_names]])
    
    # Get probability of positive class (useful)
    proba = model.predict_proba(X)[0]
    return float(proba[1]) if len(proba) > 1 else float(proba[0])


def combine_rf_and_reranking_score(
    rf_score: float,
    reranking_score: float,
    rf_weight: float = 0.7,
    reranking_weight: float = 0.3
) -> float:
    """
    Combine RF score (learned patterns) with reranking score (LLM semantic relevance).
    
    Temporary solution until RF model is retrained with reranking_score as 13th feature.
    Uses weighted average to leverage both signals.
    
    Args:
        rf_score: RF model score (0-1) - learned utility patterns from training data
        reranking_score: LLM reranking score (0-1) - semantic relevance assessment
        rf_weight: Weight for RF score (default 0.7, favors learned model)
        reranking_weight: Weight for reranking score (default 0.3)
    
    Returns:
        Combined score (0-1)
    """
    # Ensure weights sum to 1.0
    total_weight = rf_weight + reranking_weight
    if total_weight > 0:
        rf_weight = rf_weight / total_weight
        reranking_weight = reranking_weight / total_weight
    else:
        rf_weight = 0.5
        reranking_weight = 0.5
    
    # Weighted average: combines learned patterns (RF) with semantic understanding (reranking)
    combined_score = rf_weight * rf_score + reranking_weight * reranking_score
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, combined_score))


def parse_embedding(emb) -> List[float]:
    """Parse embedding from various formats (string, list, etc)."""
    if emb is None:
        return [0.0] * EMBEDDING_DIM
    if isinstance(emb, list):
        return emb
    if isinstance(emb, str):
        # Try to parse as JSON
        try:
            import json
            return json.loads(emb)
        except:
            return [0.0] * EMBEDDING_DIM
    return [0.0] * EMBEDDING_DIM


# =============================================
# Enhanced Baseline RAG Pipeline Functions
# =============================================

async def rewrite_queries(original_query: str, num_variants: int = 3) -> Tuple[List[str], int]:
    """
    Generate multiple rewritten variants of the query using LLM.
    Reduces query sensitivity and improves recall.
    
    Returns:
        Tuple of (query_variants_list, tokens_used) where tokens_used includes
        input tokens (prompt) + output tokens (generated variants)
    """
    if not ENABLE_QUERY_REWRITING:
        # Skip the round-trip; searching the original phrasing alone still works,
        # it just has slightly lower recall on awkwardly worded questions.
        return [original_query], 0

    if not providers.llm_available():
        return [original_query], 0  # Fallback to original, no tokens used

    prompt = f"""Given the following user question, generate {num_variants} different ways to ask the same question.
These should be semantically equivalent but use different wording, phrasing, or focus.

Original question: {original_query}

Generate exactly {num_variants} alternative questions, one per line. Each should be a natural, complete question.
Do not number them or add prefixes. Just list the questions:
"""

    tokens_used = 0

    try:
        text, tokens_used = await providers.llm_complete(
            prompt, max_tokens=300, temperature=0.7, timeout=60.0
        )

        # Parse lines, filter out empty and numbered items
        variants = [line.strip() for line in text.split("\n") if line.strip() and not line.strip()[0].isdigit()]
        # Remove if it starts with "Question:", "Q:", etc.
        variants = [v for v in variants if not v.lower().startswith(("question:", "q:", "alternative"))]
        variants = [v.lstrip("- ").strip() for v in variants]  # Remove bullet points
        variants = [v for v in variants if len(v) > 10]  # Filter very short items

        if variants:
            return [original_query] + variants[:num_variants], tokens_used  # Include original
    except Exception as e:
        print(f"[Rewrite] Error: {e}")

    return [original_query], tokens_used  # Fallback


async def multi_query_retrieval(
    query_variants: List[str],
    match_count_per_query: int,
    user_id: str
) -> List[dict]:
    """
    Retrieve candidates for each query variant and merge/deduplicate.
    Returns a deduplicated list of candidates with max similarity across queries.
    """
    supabase = get_supabase_client()
    all_candidates = {}  # chunk_id -> {chunk_data, max_similarity}

    # Each variant is an independent embed + vector search, so run them together.
    # Sequentially this cost one full network round-trip to Supabase per variant
    # (~1.6s each, ~16% of total question latency) for no reason -- nothing in a
    # later variant depends on an earlier one; they are merged afterwards.
    async def search_variant(query: str):
        query_embedding = await generate_embedding(query)
        if not query_embedding:
            return []
        # The Supabase client is synchronous, so keep the event loop free by
        # running the RPC in a worker thread rather than blocking on it here.
        response = await asyncio.to_thread(
            lambda: supabase.rpc(
                "match_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_count": match_count_per_query,
                    "match_user_id": user_id,
                },
            ).execute()
        )
        return response.data or []

    variant_results = await asyncio.gather(
        *(search_variant(q) for q in query_variants), return_exceptions=True
    )

    for result in variant_results:
        if isinstance(result, Exception):
            print(f"[Retrieval] variant failed: {result}")
            continue
        # Merge candidates (keep max similarity if duplicate)
        for candidate in result:
            chunk_id = candidate["id"]
            similarity = candidate.get("similarity", 0.0)

            if chunk_id not in all_candidates:
                all_candidates[chunk_id] = candidate
            elif similarity > all_candidates[chunk_id].get("similarity", 0.0):
                # A different phrasing found a better match for the same chunk
                all_candidates[chunk_id]["similarity"] = similarity


    # Convert to list and sort by similarity
    candidates = list(all_candidates.values())
    candidates.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
    
    return candidates


async def rerank_chunks(
    query: str,
    chunks: List[dict],
    top_k: int = None
) -> Tuple[List[Tuple[dict, float]], int]:
    """
    Rerank chunks using semantic scoring (LLM-based relevance).
    Returns list of (chunk, relevance_score) tuples, sorted by score descending.
    
    Returns:
        Tuple of (reranked_chunks_list, tokens_used) where tokens_used is the
        total tokens consumed (input + output) for all reranking calls
    """
    if not chunks:
        return [], 0
    
    if not providers.llm_available() or not ENABLE_LLM_RERANKING:
        # Fall back to vector ranking. Chunks keep their cosine similarity, and
        # the Random Forest still scores them downstream, so selection quality
        # degrades rather than collapsing.
        reranked = [(chunk, chunk.get("similarity", 0.0)) for chunk in chunks]
        reranked.sort(key=lambda x: x[1], reverse=True)
        if top_k:
            reranked = reranked[:top_k]
        return reranked, 0

    # Score every chunk in ONE call instead of one call per chunk.
    #
    # The old shape issued len(chunks) separate requests, capped at
    # LLM_MAX_CONCURRENCY, so 12 candidates meant two sequential waves and ~14s
    # -- the single largest slice of question latency. The scores are
    # independent, so nothing about the task required separate calls; it was
    # just how the prompt was written. Batching trades a handful of tiny
    # requests for one slightly larger one.
    #
    # Batches are capped so a large candidate pool cannot overflow the context
    # window or push the model into sloppy bookkeeping across dozens of items.
    BATCH_SIZE = 12
    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    def batch_prompt(batch: List[dict]) -> str:
        listing = "\n\n".join(
            f"[{n}]\n{c.get('content', '')[:900]}" for n, c in enumerate(batch)
        )
        return f"""You are a relevance scorer. Score how relevant each document chunk is to answering the question.

Question: {query}

{listing}

Score every chunk from 0.0 to 1.0, where:
- 1.0 = Highly relevant, directly answers the question
- 0.5 = Somewhat relevant, contains related information
- 0.0 = Not relevant at all

Reply with ONLY a JSON object mapping each chunk number to its score, for example {{"0": 0.8, "1": 0.2}}. Include every number from 0 to {len(batch) - 1}."""

    results = await providers.llm_complete_many(
        [batch_prompt(b) for b in batches],
        max_tokens=40 + 12 * BATCH_SIZE,
        temperature=0.0,
        timeout=90.0,
    )

    reranked = []
    total_tokens_used = 0

    for batch, (text, tokens) in zip(batches, results):
        total_tokens_used += tokens

        scores = {}
        if text:
            # Tolerate prose around the JSON; take the outermost braces.
            lo, hi = text.find("{"), text.rfind("}")
            if lo >= 0 and hi > lo:
                try:
                    import json as _json

                    parsed = _json.loads(text[lo:hi + 1])
                    for k, v in parsed.items():
                        idx = re.sub(r"\D", "", str(k))
                        if idx != "":
                            scores[int(idx)] = float(v)
                except (ValueError, TypeError):
                    scores = {}

        for n, chunk in enumerate(batch):
            # Cosine similarity is the fallback whenever the model omitted an
            # index or returned unparseable output, so a bad batch degrades to
            # vector ranking rather than zeroing out real candidates.
            score = chunk.get("similarity", 0.0)
            if n in scores:
                try:
                    score = max(0.0, min(1.0, float(scores[n])))
                except (ValueError, TypeError):
                    pass
            reranked.append((chunk, score))


    # Sort by relevance score descending
    reranked.sort(key=lambda x: x[1], reverse=True)

    # Return top_k if specified
    if top_k:
        reranked = reranked[:top_k]

    return reranked, total_tokens_used


def mmr_selection(
    reranked_chunks: List[Tuple[dict, float]],
    embeddings: List[List[float]],
    token_budget: int,
    lambda_param: float = 0.5
) -> List[int]:
    """
    Select chunks using Maximal Marginal Relevance (MMR).
    Balances relevance (from reranking) with diversity (low redundancy).
    
    Args:
        reranked_chunks: List of (chunk, relevance_score) tuples
        embeddings: List of chunk embeddings (same order)
        token_budget: Maximum tokens allowed
        lambda_param: Balance between relevance (1.0) and diversity (0.0). Default 0.5.
    
    Returns:
        List of selected chunk indices (from original reranked list)
    """
    if not reranked_chunks:
        return []
    
    selected = []
    remaining = list(range(len(reranked_chunks)))
    current_tokens = 0
    selected_embeddings = []
    
    # Normalize embeddings
    parsed_embeddings = [parse_embedding(e) for e in embeddings]
    
    while remaining and current_tokens < token_budget:
        best_idx = None
        best_mmr_score = -float('inf')
        
        for idx in remaining:
            chunk, relevance_score = reranked_chunks[idx]
            chunk_tokens = chunk.get("token_length", 100)
            
            # Skip if exceeds budget
            if current_tokens + chunk_tokens > token_budget:
                continue
            
            # Calculate MMR score
            # MMR = λ * relevance - (1 - λ) * max_similarity_to_selected
            relevance_term = lambda_param * relevance_score
            
            # Find max similarity to already selected chunks
            max_similarity = 0.0
            if selected_embeddings:
                chunk_emb = parsed_embeddings[idx]
                for sel_emb in selected_embeddings:
                    sim = cosine_similarity(chunk_emb, sel_emb)
                    max_similarity = max(max_similarity, sim)
            
            diversity_term = (1 - lambda_param) * max_similarity
            mmr_score = relevance_term - diversity_term
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx
        
        if best_idx is None:
            break  # No more chunks fit
        
        # Add best chunk to selection
        selected.append(best_idx)
        chunk, _ = reranked_chunks[best_idx]
        current_tokens += chunk.get("token_length", 100)
        selected_embeddings.append(parsed_embeddings[best_idx])
        remaining.remove(best_idx)
    
    return selected


async def enhanced_baseline_rag(
    query: str,
    user_id: str,
    token_budget: int = 3000,
    num_query_variants: int = 3,
    candidates_per_query: int = 10,
    rerank_top_k: int = 20,
    mmr_lambda: float = 0.5
) -> Tuple[List[dict], int, str, int]:
    """
    Enhanced baseline RAG pipeline:
    1. Query rewriting (multiple variants)
    2. Multi-query retrieval
    3. Merging and deduplication
    4. Semantic reranking
    5. MMR selection (diversity-aware)
    
    Returns: (selected_chunks, context_tokens, context_string, auxiliary_tokens)
    - context_tokens: Accurate token count of the final context string
    - auxiliary_tokens: Tokens used for query rewriting + reranking
    """
    # Step 1: Query rewriting (track tokens)
    query_variants, rewrite_tokens = await rewrite_queries(query, num_variants=num_query_variants)
    print(f"[Enhanced Baseline] Generated {len(query_variants)} query variants ({rewrite_tokens} tokens used)")
    
    # Step 2: Multi-query retrieval
    all_candidates = await multi_query_retrieval(
        query_variants=query_variants,
        match_count_per_query=candidates_per_query,
        user_id=user_id
    )
    print(f"[Enhanced Baseline] Retrieved {len(all_candidates)} unique candidates")
    
    if not all_candidates:
        return [], 0, "", rewrite_tokens  # Return auxiliary tokens even if no chunks
    
    # Step 3: Get embeddings for reranking candidates
    supabase = get_supabase_client()
    chunk_embeddings = []
    for candidate in all_candidates:
        chunk_detail = supabase.table("chunks").select("embedding").eq("id", candidate["id"]).execute()
        if chunk_detail.data and chunk_detail.data[0].get("embedding"):
            embedding = parse_embedding(chunk_detail.data[0]["embedding"])
            chunk_embeddings.append(embedding)
        else:
            chunk_embeddings.append([0.0] * EMBEDDING_DIM)
    
    # Step 4: Semantic reranking (track tokens)
    reranked, rerank_tokens = await rerank_chunks(query, all_candidates, top_k=rerank_top_k)
    print(f"[Enhanced Baseline] Reranked {len(reranked)} chunks ({rerank_tokens} tokens used)")
    
    if not reranked:
        return [], 0, "", rewrite_tokens + rerank_tokens  # Return auxiliary tokens even if no chunks
    
    # Extract reranked chunks and their scores
    reranked_chunks = [chunk for chunk, _ in reranked]
    reranked_scores = [score for _, score in reranked]
    # Map embeddings to reranked order (reranked chunks are subset of all_candidates)
    chunk_id_to_idx = {c["id"]: i for i, c in enumerate(all_candidates)}
    reranked_embeddings = []
    for c in reranked_chunks:
        idx = chunk_id_to_idx.get(c["id"])
        if idx is not None and idx < len(chunk_embeddings):
            reranked_embeddings.append(chunk_embeddings[idx])
        else:
            # Fallback: fetch embedding if somehow not in original list
            chunk_detail = supabase.table("chunks").select("embedding").eq("id", c["id"]).execute()
            if chunk_detail.data and chunk_detail.data[0].get("embedding"):
                embedding = parse_embedding(chunk_detail.data[0]["embedding"])
                reranked_embeddings.append(embedding)
            else:
                reranked_embeddings.append([0.0] * EMBEDDING_DIM)
    
    # Step 5: MMR selection
    selected_indices = mmr_selection(
        reranked_chunks=list(zip(reranked_chunks, reranked_scores)),
        embeddings=reranked_embeddings,
        token_budget=token_budget,
        lambda_param=mmr_lambda
    )
    print(f"[Enhanced Baseline] MMR selected {len(selected_indices)} chunks")
    
    # Get document filenames for context formatting
    doc_ids = list(set(c["document_id"] for c in reranked_chunks))
    docs_response = supabase.table("documents").select("id, filename").in_("id", doc_ids).execute()
    doc_filenames = {doc["id"]: doc["filename"] for doc in (docs_response.data or [])}
    
    # Build selected chunks and context
    selected_chunks = []
    context_parts = []
    
    for idx in selected_indices:
        chunk = reranked_chunks[idx]
        selected_chunks.append(chunk)
        doc_filename = doc_filenames.get(chunk["document_id"], "Unknown")
        context_parts.append(f"[Source: {doc_filename}]\n{chunk['content']}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Count context tokens accurately using Gemini's tokenizer
    context_tokens = await count_tokens_accurate(context)
    
    # Total tokens = context tokens + auxiliary operation tokens
    total_tokens = context_tokens
    auxiliary_tokens = rewrite_tokens + rerank_tokens
    
    print(f"[Enhanced Baseline] Context: {context_tokens} tokens, Auxiliary ops: {auxiliary_tokens} tokens")
    
    return selected_chunks, total_tokens, context, auxiliary_tokens


def solve_qubo_greedy(
    chunks: List[dict],
    rf_scores: List[float],
    embeddings: List[List[float]],
    token_budget: int = 3000,
    redundancy_threshold: float = 0.85,
    min_rf_score: float = 0.0
) -> List[int]:
    """
    Greedy QUBO solver for chunk selection.
    
    Objective: Maximize sum of RF scores while minimizing tokens
    Constraint: Stay within token budget
    Penalty: Redundant (highly similar) chunks
    Quality Gate: Skip chunks with RF score below min_rf_score
    
    Strategy: Select FEWER, HIGHER-QUALITY chunks (not fill budget)
    
    Returns: List of selected chunk indices
    """
    n = len(chunks)
    if n == 0:
        return []
    
    # Ensure embeddings are proper lists
    parsed_embeddings = [parse_embedding(e) for e in embeddings]
    
    # Calculate redundancy matrix (pairwise similarities)
    redundancy = np.zeros((n, n))
    try:
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(parsed_embeddings[i], parsed_embeddings[j])
                redundancy[i][j] = redundancy[j][i] = sim
    except Exception as e:
        print(f"[QUBO] Error computing redundancy matrix: {e}")
        # Return simple top-k by RF score (quality-filtered)
        sorted_indices = sorted(range(n), key=lambda i: rf_scores[i], reverse=True)
        result = []
        tokens = 0
        for idx in sorted_indices:
            if rf_scores[idx] < min_rf_score:
                continue  # Skip low quality chunks
            t = chunks[idx].get("token_length", 100) if isinstance(chunks[idx], dict) else 100
            if tokens + t <= token_budget:
                result.append(idx)
                tokens += t
        return result
    
    selected = []
    remaining = list(range(n))
    current_tokens = 0
    
    # Calculate average RF score for adaptive threshold
    avg_rf = sum(rf_scores) / len(rf_scores) if rf_scores else 0
    effective_min_rf = max(min_rf_score, avg_rf * 0.5)  # At least half of average
    
    while remaining:
        best_idx = None
        best_value = -float('inf')
        
        for idx in remaining:
            # Skip chunks below quality threshold
            if rf_scores[idx] < effective_min_rf:
                continue
                
            chunk = chunks[idx]
            chunk_tokens = chunk.get("token_length", 100) if isinstance(chunk, dict) else 100
            
            # Skip if would exceed budget
            if current_tokens + chunk_tokens > token_budget:
                continue
            
            # Calculate value: RF score minus redundancy penalty
            # Also factor in token efficiency (prefer fewer tokens for same quality)
            base_value = rf_scores[idx]
            token_efficiency = 1.0 - (chunk_tokens / token_budget) * 0.1  # Small bonus for smaller chunks
            value = base_value * token_efficiency
            
            # Penalty based on SHAP-derived weights (or defaults)
            # These values are data-driven from QUBO_WEIGHTS
            for sel_idx in selected:
                if redundancy[idx][sel_idx] > QUBO_WEIGHTS["threshold_high"]:
                    value -= redundancy[idx][sel_idx] * QUBO_WEIGHTS["penalty_high"]
                elif redundancy[idx][sel_idx] > QUBO_WEIGHTS["threshold_med"]:
                    value -= redundancy[idx][sel_idx] * QUBO_WEIGHTS["penalty_med"]
            
            if value > best_value:
                best_value = value
                best_idx = idx
        
        # Stop if no good chunks left or best value is negative (not worth adding)
        if best_idx is None or best_value < effective_min_rf * 0.5:
            break
        
        selected.append(best_idx)
        remaining.remove(best_idx)
        chunk = chunks[best_idx]
        current_tokens += chunk.get("token_length", 100) if isinstance(chunk, dict) else 100
        
        # Early stopping: if we have enough high-quality chunks, stop
        # This ensures we don't over-fill the budget with marginal chunks
        if len(selected) >= 3:
            avg_selected_rf = sum(rf_scores[i] for i in selected) / len(selected)
            # If adding more would likely decrease average quality, stop
            remaining_best_rf = max((rf_scores[i] for i in remaining if rf_scores[i] >= effective_min_rf), default=0)
            if remaining_best_rf < avg_selected_rf * 0.8:
                break  # Remaining chunks would dilute quality
    
    print(f"[QUBO] Selected {len(selected)} chunks using {current_tokens} tokens")
    return selected


def solve_chunk_selection(
    chunks: List[dict],
    rf_scores: List[float],
    embeddings: List[List[float]],
    token_budget: int,
    solver: Optional[str] = None,
) -> Tuple[List[int], dict]:
    """
    Select chunks under a token budget. Returns (indices, diagnostics).

    Dispatches on QUBO_SOLVER:
      annealing (default) -- build the real QUBO in qubo.py and minimize it with
                             simulated annealing.
      greedy              -- the original heuristic, retained so the two can be
                             benchmarked against each other rather than swapped
                             on faith.
    """
    solver = (solver or QUBO_SOLVER).strip().lower()
    if IS_EARLY_STAGE:
        # The greedy heuristic is the earlier implementation, so the early-stage
        # demo runs it for real rather than pretending the annealer is absent.
        solver = "greedy"

    if solver == "greedy":
        indices = solve_qubo_greedy(
            chunks=chunks,
            rf_scores=rf_scores,
            embeddings=embeddings,
            token_budget=token_budget,
            redundancy_threshold=0.85,
            min_rf_score=0.05,
        )
        tokens = sum(int(chunks[i].get("token_length", 100)) for i in indices)
        return indices, {"solver": "greedy", "tokens_used": tokens}

    token_lengths = [int(c.get("token_length", 100)) for c in chunks]
    parsed = [parse_embedding(e) for e in embeddings]

    result = qubo.select_chunks(
        scores=rf_scores,
        token_lengths=token_lengths,
        embeddings=parsed,
        token_budget=token_budget,
        weights=QUBO_OBJECTIVE_WEIGHTS,
        solver="annealing",
    )

    diagnostics = {
        "solver": result.solver,
        "energy": round(result.energy, 6),
        "num_variables": result.num_variables,
        "num_candidates": result.num_candidates,
        "num_slack_bits": result.num_slack_bits,
        "tokens_used": result.tokens_used,
        "weights": QUBO_OBJECTIVE_WEIGHTS.as_dict(),
        **result.diagnostics,
    }
    print(
        f"[QUBO] {result.solver}: {len(result.selected)}/{len(chunks)} chunks, "
        f"{result.tokens_used}/{token_budget} tokens, energy={result.energy:.4f}, "
        f"{result.num_variables} vars ({result.num_slack_bits} slack)"
    )
    return result.selected, diagnostics


def get_supabase_client():
    """
    Data client for backend operations.

    Returns the real Supabase client when SUPABASE_URL and a key are set,
    otherwise the local SQLite store, which implements the same call surface.
    """
    if USE_LOCAL_STORE:
        return local_store.get_local_client()

    # Use service key if available, otherwise anon key
    key = SUPABASE_SERVICE_KEY or SUPABASE_ANON_KEY
    if not key:
        raise HTTPException(status_code=500, detail="Supabase key not configured")

    return create_client(SUPABASE_URL, key)


def get_user_from_token(authorization: Optional[str] = Header(None)) -> dict:
    """
    Extract user info from Authorization header.
    Expects: Bearer <jwt_token>

    In local mode there is no Supabase Auth to validate against, so every
    request resolves to the single local identity.
    """
    if USE_LOCAL_STORE:
        return {"id": local_store.LOCAL_USER_ID, "email": local_store.LOCAL_USER_EMAIL}

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    token = authorization[7:]  # Remove "Bearer " prefix

    try:
        supabase = get_supabase_client()
        # Verify the JWT and get user
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return {"id": user_response.user.id, "email": user_response.user.email}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")


# =============================================
# Embedding Functions
# =============================================

async def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate an embedding via the active provider (see providers.py).
    Vector width is EMBEDDING_DIM. Returns None on failure.
    """
    return await providers.embed_one(text)


async def count_tokens_accurate(text: str, model: str = None) -> int:
    """
    Exact token count using the answering model's own tokenizer.

    This backs every "tokens saved" figure the comparison view reports, so it
    has to be a real tokenization rather than a character heuristic. Falls back
    to chars/4 only if the tokenizer can't be loaded.
    """
    return providers.count_tokens(text)


async def generate_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Generate embeddings for multiple texts in one batched provider call.
    """
    return await providers.embed_texts(texts)


# =============================================
# Text Extraction Functions
# =============================================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file."""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to extract PDF text: {str(e)}")


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX text: {str(e)}")


def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from plain text file with encoding detection."""
    try:
        import chardet
        detected = chardet.detect(file_bytes)
        encoding = detected.get('encoding', 'utf-8') or 'utf-8'
        return file_bytes.decode(encoding)
    except Exception as e:
        # Fallback to utf-8
        try:
            return file_bytes.decode('utf-8')
        except:
            return file_bytes.decode('latin-1')


def extract_text_from_csv(file_bytes: bytes) -> str:
    """Extract text from CSV file."""
    import csv
    try:
        import chardet
        detected = chardet.detect(file_bytes)
        encoding = detected.get('encoding', 'utf-8') or 'utf-8'
        text = file_bytes.decode(encoding)
    except:
        text = file_bytes.decode('utf-8', errors='replace')
    
    # Parse CSV and convert to readable text
    lines = []
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def extract_text_from_xlsx(file_bytes: bytes) -> str:
    """Extract text from Excel file."""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
        text_parts = []
        
        for sheet_name in wb.sheetnames:
            sheet = wb[sheet_name]
            text_parts.append(f"[Sheet: {sheet_name}]")
            for row in sheet.iter_rows(values_only=True):
                row_text = " | ".join(str(cell) if cell is not None else "" for cell in row)
                if row_text.strip():
                    text_parts.append(row_text)
            text_parts.append("")
        
        return "\n".join(text_parts)
    except Exception as e:
        raise ValueError(f"Failed to extract Excel text: {str(e)}")


def extract_text_from_json(file_bytes: bytes) -> str:
    """Extract text from JSON file."""
    import json
    try:
        data = json.loads(file_bytes.decode('utf-8'))
        return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {str(e)}")


def extract_text(file_bytes: bytes, filename: str, content_type: str) -> str:
    """
    Extract text from file based on type.
    Returns extracted text content.
    """
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if ext == 'pdf' or 'pdf' in content_type:
        return extract_text_from_pdf(file_bytes)
    elif ext in ('docx', 'doc') or 'word' in content_type:
        return extract_text_from_docx(file_bytes)
    elif ext == 'txt' or 'text/plain' in content_type:
        return extract_text_from_txt(file_bytes)
    elif ext == 'csv' or 'csv' in content_type:
        return extract_text_from_csv(file_bytes)
    elif ext in ('xlsx', 'xls') or 'spreadsheet' in content_type or 'excel' in content_type:
        return extract_text_from_xlsx(file_bytes)
    elif ext == 'json' or 'json' in content_type:
        return extract_text_from_json(file_bytes)
    else:
        # Try as plain text
        return extract_text_from_txt(file_bytes)


# =============================================
# Chunking Functions
# =============================================

def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    Simple heuristic: ~4 characters per token for English.
    """
    return max(1, len(text) // 4)


def chunk_text(text: str, max_chars: int = 1000) -> List[dict]:
    """
    Deterministic chunker: splits text into chunks of max ~1000 chars.
    No overlap. Tries to break at sentence/paragraph boundaries.
    
    Returns list of {"content": str, "token_length": int, "chunk_index": int}
    """
    if not text or not text.strip():
        return []
    
    # Clean text
    text = text.strip()
    
    # Split into paragraphs first
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    chunk_index = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If paragraph itself is too long, split by sentences
        if len(para) > max_chars:
            # Split by sentence boundaries
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # If adding this sentence exceeds limit, save current chunk
                if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chars:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "token_length": estimate_tokens(current_chunk),
                        "chunk_index": chunk_index
                    })
                    chunk_index += 1
                    current_chunk = ""
                
                # If sentence itself is too long, force split
                if len(sentence) > max_chars:
                    # Split by words
                    words = sentence.split()
                    for word in words:
                        if len(current_chunk) + len(word) + 1 > max_chars:
                            if current_chunk:
                                chunks.append({
                                    "content": current_chunk.strip(),
                                    "token_length": estimate_tokens(current_chunk),
                                    "chunk_index": chunk_index
                                })
                                chunk_index += 1
                            current_chunk = word
                        else:
                            current_chunk = f"{current_chunk} {word}".strip()
                else:
                    current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            # Paragraph fits, check if we need to start new chunk
            if current_chunk and len(current_chunk) + len(para) + 2 > max_chars:
                chunks.append({
                    "content": current_chunk.strip(),
                    "token_length": estimate_tokens(current_chunk),
                    "chunk_index": chunk_index
                })
                chunk_index += 1
                current_chunk = para
            else:
                current_chunk = f"{current_chunk}\n\n{para}".strip()
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "token_length": estimate_tokens(current_chunk),
            "chunk_index": chunk_index
        })
    
    return chunks


# =============================================
# Pydantic Models
# =============================================

class DocumentResponse(BaseModel):
    id: str
    filename: str
    relative_path: Optional[str]
    file_type: str
    created_at: str
    chunk_count: Optional[int] = None


class DocumentUploadResponse(BaseModel):
    success: bool
    documents: List[DocumentResponse]
    errors: List[str] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


# =============================================
# API Endpoints
# =============================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "stage": DEMO_STAGE,
        "status": "ok",
        "service": "Synergy RAG API",
        "version": "1.1.0",
        "rf_model_loaded": RF_MODEL is not None,
        "providers": providers.provider_status(),
        "store": local_store.stats() if USE_LOCAL_STORE else {"backend": "supabase"},
    }


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(user: dict = Depends(get_user_from_token)):
    """
    List all documents for the authenticated user.
    Returns documents with filename, relative_path, file_type, created_at.
    """
    try:
        supabase = get_supabase_client()
        
        # Fetch documents for this user
        response = supabase.table("documents").select(
            "id, filename, relative_path, file_type, created_at"
        ).eq("user_id", user["id"]).order("created_at", desc=True).execute()
        
        documents = []
        for doc in response.data or []:
            # Get chunk count for each document
            chunk_response = supabase.table("chunks").select(
                "id", count="exact"
            ).eq("document_id", doc["id"]).execute()
            
            documents.append(DocumentResponse(
                id=doc["id"],
                filename=doc["filename"],
                relative_path=doc.get("relative_path"),
                file_type=doc["file_type"],
                created_at=doc["created_at"],
                chunk_count=chunk_response.count if chunk_response.count else 0
            ))
        
        return DocumentListResponse(documents=documents, total=len(documents))
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    relative_paths: Optional[str] = Form(None),  # JSON array of paths, one per file
    user: dict = Depends(get_user_from_token)
):
    """
    Upload multiple files to the RAG knowledge base.
    
    - Uploads each file to Supabase Storage under: documents/{user_id}/{relative_path}/{filename}
    - Extracts text from each file
    - Chunks text with deterministic chunker (max ~1000 chars, no overlap)
    - Stores document metadata and chunks in database
    
    Args:
        files: List of files to upload
        relative_paths: Optional JSON array of relative paths for each file
    """
    import json
    
    # Parse relative paths if provided
    paths_list = []
    if relative_paths:
        try:
            paths_list = json.loads(relative_paths)
        except json.JSONDecodeError:
            paths_list = []
    
    # Pad paths list to match files count
    while len(paths_list) < len(files):
        paths_list.append(None)
    
    supabase = get_supabase_client()
    user_id = user["id"]
    
    uploaded_docs = []
    errors = []
    
    for i, file in enumerate(files):
        relative_path = paths_list[i]
        filename = file.filename or f"unnamed_{i}"
        
        try:
            # Read file content
            file_bytes = await file.read()
            if not file_bytes:
                errors.append(f"{filename}: Empty file")
                continue
            
            # Determine file type
            content_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
            ext = filename.lower().split('.')[-1] if '.' in filename else ''
            
            # Build storage key: documents/{user_id}/{relative_path}/{filename}
            if relative_path:
                # Clean relative path
                clean_path = relative_path.strip('/').replace('\\', '/')
                storage_key = f"{user_id}/{clean_path}/{filename}"
            else:
                storage_key = f"{user_id}/{filename}"
            
            # Upload to Supabase Storage
            try:
                upload_response = supabase.storage.from_(DOCUMENTS_BUCKET).upload(
                    path=storage_key,
                    file=file_bytes,
                    file_options={"content-type": content_type, "upsert": "true"}
                )
            except Exception as upload_error:
                errors.append(f"{filename}: Storage upload failed - {str(upload_error)}")
                continue
            
            # Extract text from file
            try:
                extracted_text = extract_text(file_bytes, filename, content_type)
            except Exception as extract_error:
                errors.append(f"{filename}: Text extraction failed - {str(extract_error)}")
                # Still save the document, but with no chunks
                extracted_text = ""
            
            # Chunk the text
            chunks = chunk_text(extracted_text)
            
            # Insert document record
            doc_data = {
                "user_id": user_id,
                "filename": filename,
                "relative_path": relative_path,
                "storage_key": storage_key,
                "storage_provider": "supabase",
                "file_type": ext or content_type.split('/')[-1]
            }
            
            doc_response = supabase.table("documents").insert(doc_data).execute()
            
            if not doc_response.data or len(doc_response.data) == 0:
                errors.append(f"{filename}: Failed to insert document record")
                continue
            
            document_id = doc_response.data[0]["id"]
            created_at = doc_response.data[0]["created_at"]
            
            # Insert chunks and generate embeddings
            if chunks:
                # First insert chunks without embeddings
                chunks_data = [
                    {
                        "document_id": document_id,
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "token_length": chunk["token_length"]
                    }
                    for chunk in chunks
                ]
                
                # Insert in batches of 100
                batch_size = 100
                inserted_chunk_ids = []
                for j in range(0, len(chunks_data), batch_size):
                    batch = chunks_data[j:j + batch_size]
                    insert_response = supabase.table("chunks").insert(batch).execute()
                    if insert_response.data:
                        inserted_chunk_ids.extend([c["id"] for c in insert_response.data])
                
                # Generate embeddings for chunks (async)
                if providers.embeddings_available() and inserted_chunk_ids:
                    chunk_texts = [chunk["content"] for chunk in chunks]
                    embeddings = await generate_embeddings_batch(chunk_texts)
                    
                    # Update chunks with embeddings
                    for idx, (chunk_id, embedding) in enumerate(zip(inserted_chunk_ids, embeddings)):
                        if embedding:
                            try:
                                supabase.table("chunks").update({
                                    "embedding": embedding
                                }).eq("id", chunk_id).execute()
                            except Exception as emb_err:
                                print(f"Failed to save embedding for chunk {chunk_id}: {emb_err}")
            
            uploaded_docs.append(DocumentResponse(
                id=document_id,
                filename=filename,
                relative_path=relative_path,
                file_type=ext or content_type.split('/')[-1],
                created_at=created_at,
                chunk_count=len(chunks)
            ))
        
        except Exception as e:
            errors.append(f"{filename}: {str(e)}")
    
    return DocumentUploadResponse(
        success=len(uploaded_docs) > 0,
        documents=uploaded_docs,
        errors=errors
    )


@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str, user: dict = Depends(get_user_from_token)):
    """
    Delete a document and its chunks.
    Also removes the file from storage.
    """
    try:
        supabase = get_supabase_client()
        
        # First, verify ownership and get storage_key
        doc_response = supabase.table("documents").select(
            "id, storage_key, user_id"
        ).eq("id", document_id).eq("user_id", user["id"]).execute()
        
        if not doc_response.data or len(doc_response.data) == 0:
            raise HTTPException(status_code=404, detail="Document not found")
        
        storage_key = doc_response.data[0]["storage_key"]
        
        # Delete from storage
        try:
            supabase.storage.from_(DOCUMENTS_BUCKET).remove([storage_key])
        except Exception as e:
            # Log but don't fail if storage deletion fails
            print(f"Warning: Failed to delete from storage: {e}")
        
        # Delete document (chunks will cascade)
        supabase.table("documents").delete().eq("id", document_id).execute()
        
        return {"success": True, "deleted_id": document_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# =============================================
# RAG Query Endpoint
# =============================================

class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 5  # Number of chunks to retrieve
    include_content: bool = True
    use_qubo: bool = True  # Use RF+QUBO selection (recommended)
    token_budget: int = 3000  # Max tokens for context
    candidate_pool: int = 20  # How many candidates to score with RF


class ChunkMatch(BaseModel):
    id: str
    document_id: str
    chunk_index: int
    content: str
    token_length: int
    similarity: float
    rf_score: Optional[float] = None  # RF model score
    document_filename: Optional[str] = None


class RAGQueryResponse(BaseModel):
    query: str
    chunks: List[ChunkMatch]
    total_tokens: int
    context: str  # Combined context for LLM
    selection_method: str = "qubo"  # "qubo" or "cosine"
    rf_model_loaded: bool = False


@app.post("/api/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest, user: dict = Depends(get_user_from_token)):
    """
    Query the RAG knowledge base with Enhanced RF+QUBO optimization.
    
    Enhanced Pipeline (if use_qubo=True):
    1. Query Rewriting (generate 3 variants)
    2. Multi-query Retrieval (retrieve from all variants, merge/deduplicate)
    3. Semantic Reranking (LLM-based relevance scoring)
    4. RF Scoring (with reranking_score as 13th feature)
    5. QUBO Selection (optimizes RF-scored chunks)
    6. Return optimized context
    
    Standard Pipeline (if use_qubo=False):
    1. Generate embedding for the query
    2. Get candidate pool via vector similarity
    3. Return top-k chunks
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not providers.embeddings_available():
        raise HTTPException(status_code=500, detail="Embedding API not configured")
    
    try:
        # Generate query embedding
        query_embedding = await generate_embedding(request.query)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        supabase = get_supabase_client()
        
        # Get larger candidate pool for RF+QUBO selection
        candidate_count = request.candidate_pool if request.use_qubo else request.top_k
        
        # ===== FILENAME-AWARE QUERYING =====
        # Extract potential filename patterns from query (e.g., "639", "639.pdf", "document.pdf")
        import re
        filename_patterns = []
        query_lower = request.query.lower()
        
        # Extract numbers, filenames with extensions, etc.
        # Patterns: standalone numbers (like "639"), filename.ext (like "639.pdf"), or just "639"
        number_pattern = r'\b(\d+)\b'  # Matches standalone numbers
        filename_ext_pattern = r'\b([\w\-_]+\.(?:pdf|docx?|txt|md|csv|xlsx?))\b'  # Matches filename.ext
        
        # Find all potential filename matches
        numbers = re.findall(number_pattern, query_lower)
        filenames = re.findall(filename_ext_pattern, query_lower)
        
        # Combine patterns (strip extensions for matching)
        all_patterns = set(numbers + [f.split('.')[0] for f in filenames] + filenames)
        
        filename_matched_chunks = []
        filename_matched_doc_ids = set()
        
        if all_patterns:
            print(f"[RAG] Detected filename patterns in query: {all_patterns}")
            # Query documents table for matching filenames
            for pattern in all_patterns:
                # Match if filename contains the pattern (case-insensitive)
                docs_response = supabase.table("documents").select(
                    "id, filename"
                ).eq("user_id", user["id"]).ilike("filename", f"%{pattern}%").execute()
                
                for doc in (docs_response.data or []):
                    filename_matched_doc_ids.add(doc["id"])
                    print(f"[RAG] Found matching document: {doc['filename']} (pattern: {pattern})")
            
            # Fetch ALL chunks from filename-matched documents
            if filename_matched_doc_ids:
                chunks_response = supabase.table("chunks").select("*").in_("document_id", list(filename_matched_doc_ids)).execute()
                
                filename_matched_chunks = chunks_response.data or []
                print(f"[RAG] Found {len(filename_matched_chunks)} chunks from filename-matched documents")
        
        # ===== SEMANTIC RETRIEVAL =====
        # Call the match_chunks function for semantic search
        response = supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": candidate_count,
                "match_user_id": user["id"]
            }
        ).execute()
        
        semantic_candidates = response.data or []
        print(f"[RAG] Found {len(semantic_candidates)} candidates from semantic search")
        
        # ===== MERGE CANDIDATES =====
        # Combine semantic and filename-matched chunks
        # Use a dict to deduplicate by chunk ID, prioritizing semantic matches (they come first)
        candidates_dict = {}
        
        # Add semantic candidates first (they get priority in case of duplicates)
        for chunk in semantic_candidates:
            candidates_dict[chunk["id"]] = chunk
        
        # Add filename-matched chunks (they might have better content matches even if semantic similarity is low)
        filename_matched_chunk_ids = set()  # Track which chunks are filename-matched
        for chunk in filename_matched_chunks:
            if chunk["id"] not in candidates_dict:
                # Set a reasonable similarity score for filename-matched chunks
                # They'll be reranked/scored properly later
                chunk["similarity"] = chunk.get("similarity", 0.5)  # Default mid-range similarity
                candidates_dict[chunk["id"]] = chunk
            filename_matched_chunk_ids.add(chunk["id"])  # Track even if already in dict
        
        candidates = list(candidates_dict.values())
        
        if not candidates:
            return RAGQueryResponse(
                query=request.query,
                chunks=[],
                total_tokens=0,
                context="",
                selection_method="none",
                rf_model_loaded=RF_MODEL is not None
            )
        
        print(f"[RAG] Total candidates after merge: {len(candidates)} (semantic: {len(semantic_candidates)}, filename-matched: {len(filename_matched_chunks)})")
        
        # Get document info for matched chunks
        doc_ids = list(set(chunk["document_id"] for chunk in candidates))
        docs_response = supabase.table("documents").select(
            "id, filename"
        ).in_("id", doc_ids).execute()
        
        doc_filenames = {doc["id"]: doc["filename"] for doc in (docs_response.data or [])}
        
        # Get total chunks per document for position calculation
        doc_chunk_counts = {}
        for doc_id in doc_ids:
            count_response = supabase.table("chunks").select(
                "id", count="exact"
            ).eq("document_id", doc_id).execute()
            doc_chunk_counts[doc_id] = count_response.count or 1
        
        # Determine selection method
        use_rf_qubo = request.use_qubo and RF_MODEL is not None
        selection_method = "qubo" if use_rf_qubo else "cosine"
        
        if use_rf_qubo:
            # ===== Enhanced RF + QUBO Pipeline: Query Rewriting + Multi-Query + Reranking + RF + QUBO =====
            # Initialize token tracking variables
            rewrite_tokens = 0
            rerank_tokens = 0
            
            try:
                # Step 1: Query Rewriting (track tokens)
                query_variants, rewrite_tokens = await rewrite_queries(request.query, num_variants=ENHANCED_BASELINE_NUM_VARIANTS)
                print(f"[RAG-QUBO] Generated {len(query_variants)} query variants ({rewrite_tokens} tokens used)")
                
                if not query_variants:
                    raise ValueError("Query rewriting returned no variants")
                
                # Step 2: Multi-query Retrieval
                all_candidates = await multi_query_retrieval(
                    query_variants=query_variants,
                    match_count_per_query=request.candidate_pool // len(query_variants) if len(query_variants) > 0 else request.candidate_pool,
                    user_id=user["id"]
                )
                print(f"[RAG-QUBO] Multi-query retrieved {len(all_candidates)} unique candidates")
                
                if not all_candidates:
                    # Fallback to single query if multi-query fails
                    print("[RAG-QUBO] Multi-query returned no candidates, using initial candidates")
                    all_candidates = candidates
                    rewrite_tokens = 0  # Reset since we're not using rewritten queries
                else:
                    # Merge multi-query results with existing candidates (which include filename-matched chunks)
                    # Use dict to deduplicate, prioritizing multi-query results
                    candidates_dict = {chunk["id"]: chunk for chunk in candidates}  # Preserve existing (includes filename-matched)
                    for chunk in all_candidates:
                        candidates_dict[chunk["id"]] = chunk  # Add/override with multi-query results
                    candidates = list(candidates_dict.values())
                    
                    print(f"[RAG-QUBO] After merge: {len(candidates)} total candidates (multi-query: {len(all_candidates)}, preserved from initial: {len(candidates) - len(all_candidates)})")
                    
                    # Re-fetch document info for new candidates
                    doc_ids = list(set(chunk["document_id"] for chunk in candidates))
                    docs_response = supabase.table("documents").select("id, filename").in_("id", doc_ids).execute()
                    doc_filenames = {doc["id"]: doc["filename"] for doc in (docs_response.data or [])}
                    doc_chunk_counts = {}
                    for doc_id in doc_ids:
                        count_response = supabase.table("chunks").select("id", count="exact").eq("document_id", doc_id).execute()
                        doc_chunk_counts[doc_id] = count_response.count or 1
                
                # Step 3: Semantic Reranking (track tokens)
                reranked, rerank_tokens = await rerank_chunks(request.query, candidates, top_k=ENHANCED_BASELINE_RERANK_TOP_K)
                print(f"[RAG-QUBO] Reranked {len(reranked)} chunks ({rerank_tokens} tokens used)")
                
                if not reranked:
                    reranked = [(c, c.get("similarity", 0.0)) for c in candidates]
                
                reranked_chunks = [chunk for chunk, _ in reranked]
                reranking_scores = {chunk["id"]: score for chunk, score in reranked}
                
                # If we have filename-matched chunks, ensure at least some make it through reranking
                # (User explicitly mentioned a filename, so prioritize those chunks)
                if filename_matched_chunk_ids:
                    reranked_chunk_ids = {chunk["id"] for chunk in reranked_chunks}
                    missing_filename_chunks = [c for c in candidates if c["id"] in filename_matched_chunk_ids and c["id"] not in reranked_chunk_ids]
                    
                    if missing_filename_chunks:
                        # Add back some filename-matched chunks that were filtered out (take first few by similarity)
                        missing_filename_chunks.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
                        num_to_add = min(3, len(missing_filename_chunks))  # Add up to 3 filename-matched chunks
                        
                        for chunk in missing_filename_chunks[:num_to_add]:
                            # Use a reasonable reranking score (mid-range) so they're considered but not dominant
                            reranking_scores[chunk["id"]] = chunk.get("similarity", 0.5)
                            reranked_chunks.append(chunk)
                        
                        print(f"[RAG-QUBO] Added {num_to_add} filename-matched chunks that were filtered out by reranking")
                
                # Step 4: RF Scoring (combine with reranking_score via weighted average)
                rf_scores = []
                chunk_embeddings = []
                
                for chunk in reranked_chunks:
                    # Get chunk embedding from database
                    chunk_detail = supabase.table("chunks").select("embedding").eq("id", chunk["id"]).execute()
                    
                    if chunk_detail.data and chunk_detail.data[0].get("embedding"):
                        embedding = parse_embedding(chunk_detail.data[0]["embedding"])
                        chunk_embeddings.append(embedding)
                    else:
                        chunk_embeddings.append([0.0] * EMBEDDING_DIM)
                    
                    # Extract features (without reranking_score - model currently has 12 features)
                    total_chunks = doc_chunk_counts.get(chunk["document_id"], 1)
                    features = extract_features_for_scoring(
                        query=request.query,
                        chunk_content=chunk["content"],
                        chunk_index=chunk["chunk_index"],
                        total_chunks=total_chunks,
                        token_length=chunk["token_length"],
                        similarity=chunk.get("similarity", 0.0)
                        # Don't pass reranking_score yet - model doesn't have this feature
                    )
                    
                    # Get RF score (12 features)
                    rf_score_base = score_chunk_with_rf(features)
                    
                    # Get reranking score
                    reranking_score = reranking_scores.get(chunk["id"], chunk.get("similarity", 0.0))
                    
                    # Combine: weighted average (70% RF, 30% reranking)
                    combined_score = combine_rf_and_reranking_score(
                        rf_score=rf_score_base,
                        reranking_score=reranking_score,
                        rf_weight=0.7,
                        reranking_weight=0.3
                    )
                    
                    rf_scores.append(combined_score)
                
                # Step 5: QUBO Selection (optimizes RF-scored chunks)
                selected_indices, qubo_diagnostics = solve_chunk_selection(
                    chunks=reranked_chunks,
                    rf_scores=rf_scores,
                    embeddings=chunk_embeddings,
                    token_budget=request.token_budget,
                )
                
                # Update candidates to reranked chunks for response building
                candidates = reranked_chunks
                
                # If QUBO returns nothing, fall back to top-k
                if not selected_indices:
                    print("[RAG-QUBO] QUBO returned no indices, falling back to top-k")
                    selected_indices = list(range(min(request.top_k, len(candidates))))
                
                print(f"[RAG-QUBO] QUBO selected {len(selected_indices)} chunks")
                
                # Build response with selected chunks
                chunks = []
                context_parts = []
                
                for idx in selected_indices:
                    if idx >= len(candidates):
                        continue
                    chunk = candidates[idx]
                    match = ChunkMatch(
                        id=chunk["id"],
                        document_id=chunk["document_id"],
                        chunk_index=chunk["chunk_index"],
                        content=chunk["content"] if request.include_content else "",
                        token_length=chunk["token_length"],
                        similarity=chunk["similarity"],
                        rf_score=rf_scores[idx] if idx < len(rf_scores) else None,
                        document_filename=doc_filenames.get(chunk["document_id"])
                    )
                    chunks.append(match)
                    
                    filename = doc_filenames.get(chunk["document_id"], "Unknown")
                    context_parts.append(f"[Source: {filename}]\n{chunk['content']}")
                
                # Count context tokens accurately
                context = "\n\n---\n\n".join(context_parts)
                context_tokens = await count_tokens_accurate(context)
                total_tokens = context_tokens
                
                print(f"[RAG-QUBO] Context: {context_tokens} tokens, Auxiliary ops: {rewrite_tokens + rerank_tokens} tokens")
            
            except Exception as qubo_error:
                import traceback
                error_trace = traceback.format_exc()
                print(f"[RAG] QUBO selection failed, falling back to cosine:")
                print(f"[RAG] Error: {str(qubo_error)}")
                print(f"[RAG] Traceback:\n{error_trace}")
                use_rf_qubo = False
                selection_method = "cosine"
                # Fall through to cosine selection below
        
        # Use cosine if QUBO is disabled or failed
        if not use_rf_qubo or selection_method == "cosine":
            # ===== Traditional Cosine Top-K =====
            
            # Just take top_k by similarity
            top_candidates = candidates[:request.top_k]
            
            chunks = []
            context_parts = []
            
            for chunk in top_candidates:
                match = ChunkMatch(
                    id=chunk["id"],
                    document_id=chunk["document_id"],
                    chunk_index=chunk["chunk_index"],
                    content=chunk["content"] if request.include_content else "",
                    token_length=chunk["token_length"],
                    similarity=chunk["similarity"],
                    rf_score=None,
                    document_filename=doc_filenames.get(chunk["document_id"])
                )
                chunks.append(match)
                
                filename = doc_filenames.get(chunk["document_id"], "Unknown")
                context_parts.append(f"[Source: {filename}]\n{chunk['content']}")
            
            # Count context tokens accurately
            context = "\n\n---\n\n".join(context_parts)
            total_tokens = await count_tokens_accurate(context)
        
        return RAGQueryResponse(
            query=request.query,
            chunks=chunks,
            total_tokens=total_tokens,
            context=context,
            selection_method=selection_method,
            rf_model_loaded=RF_MODEL is not None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")


# =============================================
# RF Model Status Endpoint
# =============================================

@app.get("/api/rag/model-status")
async def get_model_status():
    """
    Get status of the RF model and QUBO weights.
    """
    if RF_MODEL is None:
        return {
            "loaded": False,
            "message": "RF model not loaded. Using cosine similarity only.",
            "qubo_weights": QUBO_WEIGHTS,
            "weights_source": "default" if not QUBO_WEIGHTS_PATH.exists() else "shap_weights.json",
            "providers": providers.provider_status(),
            "store": local_store.stats() if USE_LOCAL_STORE else {"backend": "supabase"}
        }

    return {
        "loaded": True,
        "accuracy": RF_MODEL.get("accuracy"),
        "training_samples": RF_MODEL.get("training_samples"),
        "positive_rate": RF_MODEL.get("positive_rate"),
        "feature_names": RF_MODEL.get("feature_names"),
        "feature_importance": RF_MODEL.get("feature_importance"),
        "qubo_weights": QUBO_WEIGHTS,
        "weights_source": "default" if not QUBO_WEIGHTS_PATH.exists() else "shap_weights.json",
        "providers": providers.provider_status(),
        "store": local_store.stats() if USE_LOCAL_STORE else {"backend": "supabase"}
    }


@app.post("/api/rag/reload-model")
async def reload_model():
    """
    Reload the RF model and QUBO weights from disk.
    Useful after updating the model file or running SHAP optimization.
    """
    load_rf_model()
    load_qubo_weights()
    
    if RF_MODEL is None:
        return {"success": False, "message": "Model not found or failed to load"}
    
    return {
        "success": True,
        "accuracy": RF_MODEL.get("accuracy"),
        "qubo_weights": QUBO_WEIGHTS,
        "message": "Model and weights reloaded successfully"
    }


# =============================================
# Chat Completion Endpoint
# =============================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    context: Optional[str] = None  # Retrieved RAG context, if any
    max_tokens: int = 2048
    temperature: float = 0.4


class ChatResponse(BaseModel):
    content: str
    tokens_used: int
    model: str
    grounded: bool  # True when the answer was constrained to retrieved context


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: dict = Depends(get_user_from_token)):
    """
    Generate an assistant reply through the configured LLM provider.

    This used to be a direct browser -> Gemini call, which both exposed the API
    key to anyone opening devtools and pinned the app to a single vendor. Routing
    it here keeps the credential server-side and lets the provider swap via env.
    """
    if not providers.llm_available():
        raise HTTPException(
            status_code=503,
            detail=f"No LLM provider configured (LLM_PROVIDER={providers.LLM_PROVIDER})",
        )

    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    grounded = bool(request.context and request.context.strip())

    if grounded:
        system = (
            "You are a helpful assistant. Answer the user's question using ONLY the "
            "document context below.\n\n"
            "Rules:\n"
            "1. Use exclusively the information in DOCUMENT CONTEXT.\n"
            "2. You have no web search or outside knowledge to draw on.\n"
            "3. If the user names a file (e.g. \"what is 639.pdf about\"), find the "
            "matching [Source: ...] block in the context.\n"
            "4. If the context genuinely lacks the answer, say: \"Based on the provided "
            "documents, I cannot find information about that.\"\n\n"
            f"DOCUMENT CONTEXT:\n{request.context}"
        )
    else:
        system = "You are a helpful assistant."

    # Keep the last 12 turns, matching the previous client-side behaviour.
    transcript = "\n\n".join(
        f"{'Assistant' if m.role == 'assistant' else 'User'}: {m.content}"
        for m in request.messages[-12:]
    )
    prompt = f"{transcript}\n\nAssistant:"

    try:
        text, tokens = await providers.llm_complete(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=system,
            timeout=120.0,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {str(e)}")

    if not text:
        raise HTTPException(status_code=502, detail="LLM returned an empty response")

    return ChatResponse(
        content=text,
        tokens_used=tokens,
        model=providers.llm_model_name(),
        grounded=grounded,
    )


# =============================================
# RAG Comparison Endpoint (Traditional vs QUBO)
# =============================================

class RAGComparisonRequest(BaseModel):
    query: str
    top_k: int = 5  # For traditional RAG
    token_budget: int = 3000  # For QUBO
    candidate_pool: int = 20  # Candidates for QUBO


class ChunkInfo(BaseModel):
    id: str
    document_filename: str
    chunk_index: int
    token_length: int
    similarity: float
    rf_score: Optional[float] = None
    content_preview: str  # First 100 chars


class MethodResult(BaseModel):
    chunks_selected: int
    total_tokens: int
    avg_similarity: float
    avg_rf_score: Optional[float] = None
    chunks: List[ChunkInfo]
    redundancy_pairs: int = 0  # Number of highly similar chunk pairs


class RAGComparisonResponse(BaseModel):
    query: str
    traditional_rag: MethodResult
    qubo_rag: MethodResult
    comparison: dict  # Token savings, quality metrics, etc.


@app.post("/api/rag/compare", response_model=RAGComparisonResponse)
async def compare_rag_methods(request: RAGComparisonRequest, user: dict = Depends(get_user_from_token)):
    """
    Compare Traditional RAG vs QUBO+RF RAG side-by-side.
    
    Returns detailed comparison of:
    - Token usage
    - Chunk selection
    - Redundancy
    - Quality metrics
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    if not providers.embeddings_available():
        raise HTTPException(status_code=500, detail="Embedding API not configured")
    
    if RF_MODEL is None:
        raise HTTPException(status_code=400, detail="RF model not loaded. Cannot compare.")
    
    try:
        # Generate query embedding
        query_embedding = await generate_embedding(request.query)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        supabase = get_supabase_client()
        
        # ===== FILENAME-AWARE QUERYING (same logic as /api/rag/query) =====
        query_lower = request.query.lower()
        
        number_pattern = r'\b(\d+)\b'
        filename_ext_pattern = r'\b([\w\-_]+\.(?:pdf|docx?|txt|md|csv|xlsx?))\b'
        
        numbers = re.findall(number_pattern, query_lower)
        filenames = re.findall(filename_ext_pattern, query_lower)
        all_patterns = set(numbers + [f.split('.')[0] for f in filenames] + filenames)
        
        filename_matched_chunks = []
        filename_matched_doc_ids = set()
        
        if all_patterns:
            print(f"[Compare] Detected filename patterns in query: {all_patterns}")
            for pattern in all_patterns:
                docs_response = supabase.table("documents").select(
                    "id, filename"
                ).eq("user_id", user["id"]).ilike("filename", f"%{pattern}%").execute()
                
                for doc in (docs_response.data or []):
                    filename_matched_doc_ids.add(doc["id"])
            
            if filename_matched_doc_ids:
                chunks_response = supabase.table("chunks").select("*").in_("document_id", list(filename_matched_doc_ids)).execute()
                filename_matched_chunks = chunks_response.data or []
        
        # ===== SEMANTIC RETRIEVAL =====
        response = supabase.rpc(
            "match_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": request.candidate_pool,
                "match_user_id": user["id"]
            }
        ).execute()
        
        semantic_candidates = response.data or []
        
        # ===== MERGE CANDIDATES =====
        candidates_dict = {}
        for chunk in semantic_candidates:
            candidates_dict[chunk["id"]] = chunk
        
        for chunk in filename_matched_chunks:
            if chunk["id"] not in candidates_dict:
                chunk["similarity"] = chunk.get("similarity", 0.5)
                candidates_dict[chunk["id"]] = chunk
        
        candidates = list(candidates_dict.values())
        
        if not candidates:
            raise HTTPException(status_code=404, detail="No documents found")
        
        # Get document info
        doc_ids = list(set(chunk["document_id"] for chunk in candidates))
        docs_response = supabase.table("documents").select("id, filename").in_("id", doc_ids).execute()
        doc_filenames = {doc["id"]: doc["filename"] for doc in (docs_response.data or [])}
        
        # Get chunk counts per document
        doc_chunk_counts = {}
        for doc_id in doc_ids:
            count_response = supabase.table("chunks").select("id", count="exact").eq("document_id", doc_id).execute()
            doc_chunk_counts[doc_id] = count_response.count or 1
        
        # ===== ENHANCED BASELINE RAG: Multi-query + Reranking + MMR =====
        enhanced_baseline_chunks, enhanced_baseline_context_tokens, enhanced_context, enhanced_auxiliary_tokens = await enhanced_baseline_rag(
            query=request.query,
            user_id=user["id"],
            token_budget=request.token_budget if request.token_budget else 3000,
            num_query_variants=ENHANCED_BASELINE_NUM_VARIANTS,
            candidates_per_query=ENHANCED_BASELINE_CANDIDATES_PER_QUERY,
            rerank_top_k=ENHANCED_BASELINE_RERANK_TOP_K,
            mmr_lambda=ENHANCED_BASELINE_MMR_LAMBDA
        )
        
        # Total tokens = context tokens + auxiliary operation tokens (query rewriting + reranking)
        traditional_tokens = enhanced_baseline_context_tokens + enhanced_auxiliary_tokens
        
        # Map to ChunkInfo format for comparison
        traditional_chunks = []
        traditional_similarities = []
        traditional_rf_scores = []
        
        # Compute RF scores for enhanced baseline chunks
        for chunk in enhanced_baseline_chunks:
            # Extract features and compute RF score
            total_chunks = doc_chunk_counts.get(chunk["document_id"], 1)
            features = extract_features_for_scoring(
                query=request.query,
                chunk_content=chunk["content"],
                chunk_index=chunk["chunk_index"],
                total_chunks=total_chunks,
                token_length=chunk["token_length"],
                similarity=chunk.get("similarity", 0.0)
            )
            rf_score = score_chunk_with_rf(features)
            
            traditional_chunks.append(ChunkInfo(
                id=chunk["id"],
                document_filename=doc_filenames.get(chunk["document_id"], "Unknown"),
                chunk_index=chunk["chunk_index"],
                token_length=chunk["token_length"],
                similarity=chunk.get("similarity", 0.0),
                rf_score=rf_score,
                content_preview=chunk["content"][:100] + "..."
            ))
            traditional_similarities.append(chunk.get("similarity", 0.0))
            traditional_rf_scores.append(rf_score)
        
        # Count redundant pairs in enhanced baseline selection
        traditional_redundancy = 0
        try:
            # Get embeddings for enhanced baseline chunks
            enhanced_embeddings = []
            for chunk in enhanced_baseline_chunks:
                chunk_detail = supabase.table("chunks").select("embedding").eq("id", chunk["id"]).execute()
                if chunk_detail.data and chunk_detail.data[0].get("embedding"):
                    embedding = parse_embedding(chunk_detail.data[0]["embedding"])
                    enhanced_embeddings.append(embedding)
                else:
                    enhanced_embeddings.append([0.0] * EMBEDDING_DIM)
            
            for i in range(len(enhanced_embeddings)):
                for j in range(i + 1, len(enhanced_embeddings)):
                    emb_i = parse_embedding(enhanced_embeddings[i])
                    emb_j = parse_embedding(enhanced_embeddings[j])
                    sim = cosine_similarity(emb_i, emb_j)
                    if sim > 0.85:
                        traditional_redundancy += 1
        except Exception as e:
            print(f"[Compare] Error computing enhanced baseline redundancy: {e}")
        
        # ===== QUBO+RF RAG: Enhanced Pipeline (Query Rewriting + Multi-Query + Reranking + RF + QUBO) =====
        # Use traditional tokens as budget for fair comparison (same constraint)
        qubo_budget = traditional_tokens if traditional_tokens > 0 else request.token_budget
        
        # Step 1: Query Rewriting (same as enhanced baseline for fair comparison, track tokens)
        query_variants, qubo_rewrite_tokens = await rewrite_queries(request.query, num_variants=ENHANCED_BASELINE_NUM_VARIANTS)
        print(f"[Compare-QUBO] Generated {len(query_variants)} query variants ({qubo_rewrite_tokens} tokens used)")
        
        # Step 2: Multi-query Retrieval
        qubo_candidates = await multi_query_retrieval(
            query_variants=query_variants,
            match_count_per_query=request.candidate_pool // len(query_variants) if len(query_variants) > 0 else request.candidate_pool,
            user_id=user["id"]
        )
        print(f"[Compare-QUBO] Multi-query retrieved {len(qubo_candidates)} unique candidates")
        
        if not qubo_candidates:
            qubo_candidates = candidates  # Fallback
        
        # Step 3: Semantic Reranking (track tokens)
        reranked, qubo_rerank_tokens = await rerank_chunks(request.query, qubo_candidates, top_k=ENHANCED_BASELINE_RERANK_TOP_K)
        print(f"[Compare-QUBO] Reranked {len(reranked)} chunks ({qubo_rerank_tokens} tokens used)")
        
        if not reranked:
            reranked = [(c, c.get("similarity", 0.0)) for c in qubo_candidates]
        
        reranked_chunks = [chunk for chunk, _ in reranked]
        reranking_scores = {chunk["id"]: score for chunk, score in reranked}
        
        # Update document metadata if new documents found
        qubo_doc_ids = set(chunk["document_id"] for chunk in reranked_chunks)
        new_doc_ids = qubo_doc_ids - set(doc_ids)
        if new_doc_ids:
            new_docs_response = supabase.table("documents").select("id, filename").in_("id", list(new_doc_ids)).execute()
            for doc in (new_docs_response.data or []):
                doc_filenames[doc["id"]] = doc["filename"]
            for doc_id in new_doc_ids:
                count_response = supabase.table("chunks").select("id", count="exact").eq("document_id", doc_id).execute()
                doc_chunk_counts[doc_id] = count_response.count or 1
        
        # Get embeddings for reranked candidates
        qubo_chunk_embeddings = []
        for chunk in reranked_chunks:
            chunk_detail = supabase.table("chunks").select("embedding").eq("id", chunk["id"]).execute()
            if chunk_detail.data and chunk_detail.data[0].get("embedding"):
                embedding = parse_embedding(chunk_detail.data[0]["embedding"])
                qubo_chunk_embeddings.append(embedding)
            else:
                qubo_chunk_embeddings.append([0.0] * EMBEDDING_DIM)
        
        # Step 4: RF Scoring (combine with reranking_score via weighted average)
        qubo_rf_scores = []
        for chunk in reranked_chunks:
            total_chunks = doc_chunk_counts.get(chunk["document_id"], 1)
            
            # Extract features (without reranking_score - model currently has 12 features)
            features = extract_features_for_scoring(
                query=request.query,
                chunk_content=chunk["content"],
                chunk_index=chunk["chunk_index"],
                total_chunks=total_chunks,
                token_length=chunk["token_length"],
                similarity=chunk.get("similarity", 0.0)
                # Don't pass reranking_score yet - model doesn't have this feature
            )
            
            # Get RF score (12 features)
            rf_score_base = score_chunk_with_rf(features)
            
            # Get reranking score
            reranking_score = reranking_scores.get(chunk["id"], chunk.get("similarity", 0.0))
            
            # Combine: weighted average (70% RF, 30% reranking)
            combined_score = combine_rf_and_reranking_score(
                rf_score=rf_score_base,
                reranking_score=reranking_score,
                rf_weight=0.7,
                reranking_weight=0.3
            )
            
            qubo_rf_scores.append(combined_score)
        
        # Step 5: QUBO Selection
        qubo_indices, qubo_diagnostics = solve_chunk_selection(
            chunks=reranked_chunks,
            rf_scores=qubo_rf_scores,
            embeddings=qubo_chunk_embeddings,
            token_budget=qubo_budget,
        )
        
        qubo_chunks = []
        qubo_context_parts = []
        qubo_similarities = []
        qubo_combined_scores = []
        
        for idx in qubo_indices:
            chunk = reranked_chunks[idx]
            qubo_chunks.append(ChunkInfo(
                id=chunk["id"],
                document_filename=doc_filenames.get(chunk["document_id"], "Unknown"),
                chunk_index=chunk["chunk_index"],
                token_length=chunk["token_length"],
                similarity=chunk.get("similarity", 0.0),
                rf_score=qubo_rf_scores[idx],  # RF score (includes reranking_score as 13th feature)
                content_preview=chunk["content"][:100] + "..."
            ))
            qubo_context_parts.append(f"[Source: {doc_filenames.get(chunk['document_id'], 'Unknown')}]\n{chunk['content']}")
            qubo_similarities.append(chunk.get("similarity", 0.0))
            qubo_combined_scores.append(qubo_rf_scores[idx])
        
        # Count context tokens accurately
        qubo_context = "\n\n---\n\n".join(qubo_context_parts)
        qubo_context_tokens = await count_tokens_accurate(qubo_context)
        
        # Total tokens = context tokens + auxiliary operation tokens (query rewriting + reranking)
        qubo_tokens = qubo_context_tokens + qubo_rewrite_tokens + qubo_rerank_tokens
        
        print(f"[Compare-QUBO] Context: {qubo_context_tokens} tokens, Auxiliary ops: {qubo_rewrite_tokens + qubo_rerank_tokens} tokens, Total: {qubo_tokens} tokens")
        
        # Count redundant pairs in QUBO selection
        qubo_redundancy = 0
        try:
            for i in range(len(qubo_indices)):
                for j in range(i + 1, len(qubo_indices)):
                    emb_i = parse_embedding(qubo_chunk_embeddings[qubo_indices[i]])
                    emb_j = parse_embedding(qubo_chunk_embeddings[qubo_indices[j]])
                    sim = cosine_similarity(emb_i, emb_j)
                    if sim > 0.85:
                        qubo_redundancy += 1
        except Exception as e:
            print(f"[Compare] Error computing QUBO redundancy: {e}")
        
        # ===== BUILD RESULTS =====
        traditional_result = MethodResult(
            chunks_selected=len(traditional_chunks),
            total_tokens=traditional_tokens,
            avg_similarity=sum(traditional_similarities) / len(traditional_similarities) if traditional_similarities else 0,
            avg_rf_score=sum(traditional_rf_scores) / len(traditional_rf_scores) if traditional_rf_scores else 0,
            chunks=traditional_chunks,
            redundancy_pairs=traditional_redundancy
        )
        
        qubo_result = MethodResult(
            chunks_selected=len(qubo_chunks),
            total_tokens=qubo_tokens,
            avg_similarity=sum(qubo_similarities) / len(qubo_similarities) if qubo_similarities else 0,
            avg_rf_score=sum(qubo_combined_scores) / len(qubo_combined_scores) if qubo_combined_scores else 0,
            chunks=qubo_chunks,
            redundancy_pairs=qubo_redundancy
        )
        
        # ===== COMPARISON METRICS =====
        token_savings = traditional_tokens - qubo_tokens
        token_savings_pct = (token_savings / traditional_tokens * 100) if traditional_tokens > 0 else 0
        
        rf_score_improvement = qubo_result.avg_rf_score - traditional_result.avg_rf_score
        rf_score_improvement_pct = (rf_score_improvement / traditional_result.avg_rf_score * 100) if traditional_result.avg_rf_score > 0 else 0
        
        redundancy_reduction = traditional_redundancy - qubo_redundancy
        
        # Calculate efficiency score (quality per token)
        trad_efficiency = (traditional_result.avg_rf_score / traditional_tokens * 1000) if traditional_tokens > 0 else 0
        qubo_efficiency = (qubo_result.avg_rf_score / qubo_tokens * 1000) if qubo_tokens > 0 else 0
        efficiency_improvement = qubo_efficiency - trad_efficiency
        
        # ----- Verdict -----
        #
        # This used to be a points tally: +2 for saving tokens, +2 for higher
        # relevance, +1 each for redundancy and efficiency, and >=3 printed
        # "significantly outperforms". That let the banner declare a decisive win
        # on token savings alone while average relevance had actually FALLEN --
        # which is the first thing a reviewer asks about.
        #
        # Two facts make a tally the wrong tool here:
        #
        #   1. avg_rf_score is a MEAN. Dropping a high-scoring near-duplicate for
        #      a distinct lower-scoring chunk lowers it while improving the
        #      context; keeping only the single best chunk raises it while making
        #      the answer worse. It moves for reasons unrelated to answer quality,
        #      in both directions.
        #   2. Fewer tokens at lower relevance is a TRADE, not a win.
        #
        # So state the trade instead of scoring it. Token savings is measured and
        # trustworthy; relevance is directional and always reported with its sign.
        RELEVANCE_TOLERANCE_PCT = 5.0  # within +/-5% relative is "comparable"

        saved_pct = abs(round(token_savings_pct))
        rel_pct = rf_score_improvement_pct

        if IS_EARLY_STAGE:
            # The early milestone reports only what it actually established:
            # context size. Relevance framing and the trade-off wording came out
            # of later measurement work, so surfacing them here would overstate
            # where the project was. Still a true statement, just a smaller one.
            overall = (
                f"{abs(round(token_savings_pct))}% smaller context"
                if token_savings > 0 else "No token saving"
            )
            verdict_kind = "good" if token_savings > 0 else "neutral"
        elif token_savings <= 0:
            overall = "No token saving"
        elif rel_pct > RELEVANCE_TOLERANCE_PCT:
            overall = f"{saved_pct}% fewer tokens, higher relevance"
        elif rel_pct >= -RELEVANCE_TOLERANCE_PCT:
            overall = f"{saved_pct}% fewer tokens at comparable relevance"
        else:
            # Name the cost explicitly rather than burying it in a green banner.
            overall = f"{saved_pct}% fewer tokens, {abs(round(rel_pct))}% lower relevance"

        # Drives banner styling on the client: a trade is not styled as a win.
        if not IS_EARLY_STAGE:
            verdict_kind = (
                "neutral" if token_savings <= 0
                else "win" if rel_pct > RELEVANCE_TOLERANCE_PCT
                else "good" if rel_pct >= -RELEVANCE_TOLERANCE_PCT
                else "tradeoff"
            )

        
        comparison = {
            "token_savings": {
                "absolute": token_savings,
                "percentage": round(token_savings_pct, 1),
                "traditional_tokens": traditional_tokens,
                "qubo_tokens": qubo_tokens,
                "verdict": f"Saved {abs(token_savings)} tokens ({abs(round(token_savings_pct, 1))}%)" if token_savings > 0 else "Same usage" if token_savings == 0 else "Slight increase"
            },
            "quality": {
                "rf_score_improvement": round(rf_score_improvement, 4),
                "rf_score_improvement_pct": round(rf_score_improvement_pct, 1),
                "traditional_avg_rf": round(traditional_result.avg_rf_score, 4),
                "qubo_avg_rf": round(qubo_result.avg_rf_score, 4),
                # Report direction honestly. A meaningful drop previously read
                # "Similar quality", which is how a regression gets shipped.
                "verdict": (
                    f"+{round(rf_score_improvement_pct, 1)}% relevance"
                    if rf_score_improvement_pct > 5
                    else f"{round(rf_score_improvement_pct, 1)}% relevance"
                    if rf_score_improvement_pct < -5
                    else "Comparable relevance"
                ),
                # avg_rf_score is a mean, so it falls when a high-scoring duplicate
                # is swapped for a distinct lower-scoring chunk. That is the
                # redundancy term working -- but it is not a win either, which is
                # why it is never scored as one.
                "note": (
                    "Lower average relevance is expected when redundant chunks are "
                    "dropped for diverse ones: a trade, not an improvement."
                    if (rf_score_improvement_pct < -5 and not IS_EARLY_STAGE) else None
                )
            },
            "redundancy": {
                "traditional_redundant_pairs": traditional_redundancy,
                "qubo_redundant_pairs": qubo_redundancy,
                "reduction": redundancy_reduction,
                "verdict": f"-{redundancy_reduction} redundant pairs" if redundancy_reduction > 0 else "Clean context" if traditional_redundancy == 0 else "Same"
            },
            "efficiency": {
                "traditional_chunks": len(traditional_chunks),
                "qubo_chunks": len(qubo_chunks),
                "tokens_per_chunk_traditional": round(traditional_tokens / len(traditional_chunks), 1) if traditional_chunks else 0,
                "tokens_per_chunk_qubo": round(qubo_tokens / len(qubo_chunks), 1) if qubo_chunks else 0,
                "quality_per_1k_tokens_traditional": round(trad_efficiency, 2),
                "quality_per_1k_tokens_qubo": round(qubo_efficiency, 2)
            },
            "overall_verdict": overall,
            "verdict_kind": verdict_kind,
            "stage": DEMO_STAGE,
            # Name the selector that actually produced this column. Hardcoding
            # "QUBO+RF" in the UI made the panel assert something untrue whenever
            # a different selector ran -- the label has to follow the code path.
            "selector_label": (
                "Greedy Selection" if (qubo_diagnostics or {}).get("solver") == "greedy"
                else "QUBO + Annealing"
            ),
            # Solver internals (energy, variable count, weights, any budget
            # repairs) so a reported win can be traced back to what ran.
            # Solver internals are a later-milestone artifact; the early stage
            # omits the key entirely rather than sending a redacted version, so
            # the client simply has nothing to render.
            "qubo_solver": None if IS_EARLY_STAGE else qubo_diagnostics
        }
        
        return RAGComparisonResponse(
            query=request.query,
            traditional_rag=traditional_result,
            qubo_rag=qubo_result,
            comparison=comparison
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


# =============================================
# Re-embed existing chunks (utility endpoint)
# =============================================

@app.post("/api/documents/reembed")
async def reembed_documents(user: dict = Depends(get_user_from_token)):
    """
    Re-generate embeddings for all chunks belonging to the user.
    Useful after enabling embeddings for existing documents.
    """
    if not providers.embeddings_available():
        raise HTTPException(status_code=500, detail="Embedding API not configured")
    
    try:
        supabase = get_supabase_client()
        
        # Get all chunks for user's documents that don't have embeddings
        response = supabase.table("chunks").select(
            "id, content, document_id"
        ).is_("embedding", "null").execute()
        
        # Filter to only user's documents
        if response.data:
            doc_ids = list(set(c["document_id"] for c in response.data))
            docs_response = supabase.table("documents").select("id").eq(
                "user_id", user["id"]
            ).in_("id", doc_ids).execute()
            
            user_doc_ids = set(d["id"] for d in (docs_response.data or []))
            chunks_to_embed = [c for c in response.data if c["document_id"] in user_doc_ids]
        else:
            chunks_to_embed = []
        
        embedded_count = 0
        errors = []
        
        for chunk in chunks_to_embed:
            try:
                embedding = await generate_embedding(chunk["content"])
                if embedding:
                    supabase.table("chunks").update({
                        "embedding": embedding
                    }).eq("id", chunk["id"]).execute()
                    embedded_count += 1
            except Exception as e:
                errors.append(f"Chunk {chunk['id']}: {str(e)}")
        
        return {
            "success": True,
            "embedded_count": embedded_count,
            "total_chunks": len(chunks_to_embed),
            "errors": errors[:10]  # Limit error list
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Re-embedding failed: {str(e)}")


# =============================================
# Run with: uvicorn app:app --reload --port 8001
# =============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)

