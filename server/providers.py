"""
Pluggable providers for embeddings, LLM completion, and token counting.

The original build hardwired Google Gemini for all three. That key is dead and
Gemini billing is depleted, so the defaults here are the things that actually
work on this machine:

  * embeddings  -> local ONNX (fastembed / BAAI-bge-small-en-v1.5, 384-dim).
                   No network after the first model download, no quota.
  * completion  -> Yotta / Shakti Cloud, an OpenAI-compatible gateway.
  * tokens      -> the real DeepSeek-V3 tokenizer, downloaded once from HF.

Every provider is selected by env var so a working Gemini key can be dropped
back in later without touching call sites:

  EMBEDDING_PROVIDER = local | gemini
  LLM_PROVIDER       = yotta | gemini | none
"""

import os
import asyncio
from typing import List, Optional, Tuple

import httpx

# =============================================
# Configuration
# =============================================

EMBEDDING_PROVIDER = (os.getenv("EMBEDDING_PROVIDER") or "local").strip().lower()
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "yotta").strip().lower()

# --- Local embedding model ---
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
LOCAL_EMBEDDING_DIM = int(os.getenv("LOCAL_EMBEDDING_DIM") or "384")

# --- Gemini (optional fallback) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("VITE_GEMINI_API_KEY")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL") or "text-embedding-004"
GEMINI_EMBEDDING_DIM = 768
GEMINI_GENERATION_MODEL = os.getenv("GEMINI_MODEL") or "gemini-2.0-flash"

# --- Yotta / Shakti Cloud (OpenAI-compatible) ---
YOTTA_BASE_URL = (os.getenv("YOTTA_BASE_URL") or "").rstrip("/")
YOTTA_API_KEY = os.getenv("YOTTA_API_KEY") or ""
YOTTA_GATEWAY_ID = os.getenv("YOTTA_GATEWAY_ID") or ""
YOTTA_MODEL = os.getenv("YOTTA_MODEL") or "deepseek-v3"
# The `id` header pins the request to ONE deployment, so it must only be sent
# for that deployment's model. Other models route by name and 400 if it's set.
YOTTA_GATEWAY_MODEL = os.getenv("YOTTA_GATEWAY_MODEL") or "deepseek-v3"
YOTTA_THINKING = (os.getenv("YOTTA_THINKING") or "disabled").strip().lower()

# How many rerank/score calls may be in flight at once. The original code did
# these strictly sequentially, which put a 20-chunk rerank at ~60s of pure
# round-trip latency and made the compare endpoint look hung.
LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY") or "8")

# --- Token counting ---
TOKENIZER_REPO = os.getenv("TOKENIZER_REPO") or "deepseek-ai/DeepSeek-V3"


# =============================================
# Embeddings
# =============================================

_local_embedder = None
_local_embedder_failed = False


def embedding_dim() -> int:
    """Vector width of the active embedding provider."""
    if EMBEDDING_PROVIDER == "gemini":
        return GEMINI_EMBEDDING_DIM
    return LOCAL_EMBEDDING_DIM


def _get_local_embedder():
    """Lazily construct the ONNX embedder (first call downloads the model)."""
    global _local_embedder, _local_embedder_failed
    if _local_embedder is not None or _local_embedder_failed:
        return _local_embedder
    try:
        from fastembed import TextEmbedding

        _local_embedder = TextEmbedding(model_name=LOCAL_EMBEDDING_MODEL)
        print(f"[Embed] Local ONNX embedder ready: {LOCAL_EMBEDDING_MODEL} ({LOCAL_EMBEDDING_DIM}d)")
    except Exception as e:
        _local_embedder_failed = True
        print(f"[Embed] Failed to load local embedder: {e}")
    return _local_embedder


def embeddings_available() -> bool:
    if EMBEDDING_PROVIDER == "gemini":
        return bool(GEMINI_API_KEY)
    return _get_local_embedder() is not None


def _embed_local_sync(texts: List[str]) -> List[Optional[List[float]]]:
    """Blocking ONNX inference over a batch. Called via a worker thread."""
    embedder = _get_local_embedder()
    if embedder is None:
        return [None] * len(texts)
    try:
        vectors = list(embedder.embed(texts))
        return [[float(x) for x in v] for v in vectors]
    except Exception as e:
        print(f"[Embed] Local embedding error: {e}")
        return [None] * len(texts)


async def _embed_gemini_one(text: str) -> Optional[List[float]]:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_EMBEDDING_MODEL}:embedContent?key={GEMINI_API_KEY}"
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                json={
                    "model": f"models/{GEMINI_EMBEDDING_MODEL}",
                    "content": {"parts": [{"text": text}]},
                },
            )
        if response.status_code != 200:
            print(f"[Embed] Gemini error {response.status_code}: {response.text[:200]}")
            return None
        values = response.json().get("embedding", {}).get("values", [])
        return values or None
    except Exception as e:
        print(f"[Embed] Gemini request failed: {e}")
        return None


async def embed_texts(texts: List[str]) -> List[Optional[List[float]]]:
    """Embed a batch of texts. Returns None in a slot that failed."""
    if not texts:
        return []

    # Blank input can't be embedded; keep slot alignment with the caller's list.
    cleaned = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    results: List[Optional[List[float]]] = [None] * len(texts)
    if not cleaned:
        return results

    # Both providers degrade badly on very long input.
    max_chars = 25000
    payload = [t[:max_chars] for _, t in cleaned]

    if EMBEDDING_PROVIDER == "gemini":
        vectors = await asyncio.gather(*[_embed_gemini_one(t) for t in payload])
    else:
        # ONNX inference is CPU-bound; keep it off the event loop.
        vectors = await asyncio.to_thread(_embed_local_sync, payload)

    expected = embedding_dim()
    for (slot, _), vector in zip(cleaned, vectors):
        if vector and len(vector) == expected:
            results[slot] = vector
        elif vector:
            print(f"[Embed] Unexpected dimension {len(vector)} (expected {expected})")
    return results


async def embed_one(text: str) -> Optional[List[float]]:
    """Embed a single text."""
    return (await embed_texts([text]))[0]


# =============================================
# Token counting
# =============================================

_tokenizer = None
_tokenizer_failed = False


def _get_tokenizer():
    """Real tokenizer for the answering model, so token deltas aren't guesses."""
    global _tokenizer, _tokenizer_failed
    if _tokenizer is not None or _tokenizer_failed:
        return _tokenizer
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        path = hf_hub_download(repo_id=TOKENIZER_REPO, filename="tokenizer.json")
        _tokenizer = Tokenizer.from_file(path)
        print(f"[Tokens] Tokenizer ready: {TOKENIZER_REPO}")
    except Exception as e:
        _tokenizer_failed = True
        print(f"[Tokens] Tokenizer unavailable ({e}); falling back to char heuristic")
    return _tokenizer


def estimate_tokens_heuristic(text: str) -> int:
    """Last-resort approximation: ~4 characters per token."""
    return max(1, len(text) // 4) if text else 0


def count_tokens(text: str) -> int:
    """Exact token count via the model tokenizer, heuristic if unavailable."""
    if not text or not text.strip():
        return 0
    tokenizer = _get_tokenizer()
    if tokenizer is None:
        return estimate_tokens_heuristic(text)
    try:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)
    except Exception:
        return estimate_tokens_heuristic(text)


def tokenizer_name() -> str:
    return TOKENIZER_REPO if _get_tokenizer() is not None else "heuristic(chars/4)"


# =============================================
# LLM completion
# =============================================

def llm_available() -> bool:
    if LLM_PROVIDER == "yotta":
        return bool(YOTTA_BASE_URL and YOTTA_API_KEY)
    if LLM_PROVIDER == "gemini":
        return bool(GEMINI_API_KEY)
    return False


def llm_model_name() -> str:
    if LLM_PROVIDER == "yotta":
        return YOTTA_MODEL
    if LLM_PROVIDER == "gemini":
        return GEMINI_GENERATION_MODEL
    return "none"


async def _complete_yotta(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system: Optional[str],
    timeout: float,
) -> Tuple[str, int]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {YOTTA_API_KEY}",
    }
    if YOTTA_GATEWAY_ID and YOTTA_MODEL == YOTTA_GATEWAY_MODEL:
        headers["id"] = YOTTA_GATEWAY_ID

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": YOTTA_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    # DeepSeek defaults to thinking mode, which bills reasoning tokens against
    # max_tokens and can starve the actual answer.
    if YOTTA_MODEL.startswith("deepseek"):
        body["thinking"] = {"type": YOTTA_THINKING}

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{YOTTA_BASE_URL}/chat/completions", headers=headers, json=body
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Yotta chat/completions failed: {response.status_code} {response.text[:300]}"
        )

    data = response.json()
    choices = data.get("choices") or []
    text = ""
    if choices:
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        # Reasoning models sometimes return only reasoning_content on truncation.
        if not text:
            text = message.get("reasoning_content") or ""

    usage = data.get("usage") or {}
    tokens = usage.get("total_tokens")
    if not tokens:
        tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
    if not tokens:
        tokens = count_tokens(prompt) + count_tokens(text)

    return text.strip(), int(tokens)


async def _complete_gemini(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system: Optional[str],
    timeout: float,
) -> Tuple[str, int]:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_GENERATION_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature},
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=body)

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini generateContent failed: {response.status_code} {response.text[:300]}"
        )

    data = response.json()
    text = ""
    try:
        text = data["candidates"][0]["content"]["parts"][0].get("text", "")
    except (KeyError, IndexError):
        pass

    usage = data.get("usageMetadata") or {}
    tokens = usage.get("promptTokenCount", 0) + usage.get("candidatesTokenCount", 0)
    if not tokens:
        tokens = count_tokens(prompt) + count_tokens(text)

    return text.strip(), int(tokens)


async def llm_complete(
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
    system: Optional[str] = None,
    timeout: float = 60.0,
) -> Tuple[str, int]:
    """
    Single completion.

    Returns (text, tokens_used) where tokens_used is the provider-reported
    input+output total. Raises on transport/provider failure so callers can
    decide whether to fall back.
    """
    if not llm_available():
        return "", 0
    if LLM_PROVIDER == "yotta":
        return await _complete_yotta(prompt, max_tokens, temperature, system, timeout)
    if LLM_PROVIDER == "gemini":
        return await _complete_gemini(prompt, max_tokens, temperature, system, timeout)
    return "", 0


async def llm_complete_many(
    prompts: List[str],
    max_tokens: int = 32,
    temperature: float = 0.0,
    system: Optional[str] = None,
    timeout: float = 60.0,
) -> List[Tuple[Optional[str], int]]:
    """
    Run many small completions concurrently, bounded by LLM_MAX_CONCURRENCY.

    Returns one (text_or_None, tokens_used) per prompt, in order. A failed call
    yields (None, 0) rather than raising, so one bad chunk can't sink a rerank.
    """
    if not prompts or not llm_available():
        return [(None, 0)] * len(prompts)

    semaphore = asyncio.Semaphore(max(1, LLM_MAX_CONCURRENCY))

    async def run(prompt: str) -> Tuple[Optional[str], int]:
        async with semaphore:
            try:
                text, tokens = await llm_complete(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system,
                    timeout=timeout,
                )
                return text, tokens
            except Exception as e:
                print(f"[LLM] Call failed: {e}")
                return None, 0

    return await asyncio.gather(*[run(p) for p in prompts])


def provider_status() -> dict:
    """Snapshot for the /api/rag/model-status endpoint."""
    return {
        "embedding_provider": EMBEDDING_PROVIDER,
        "embedding_model": (
            GEMINI_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "gemini" else LOCAL_EMBEDDING_MODEL
        ),
        "embedding_dim": embedding_dim(),
        "embeddings_available": embeddings_available(),
        "llm_provider": LLM_PROVIDER,
        "llm_model": llm_model_name(),
        "llm_available": llm_available(),
        "tokenizer": tokenizer_name(),
    }
