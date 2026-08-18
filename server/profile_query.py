"""
Where does the latency of one question actually go?

Times each stage of the pipeline separately so optimization targets the real
bottleneck instead of the suspected one. Run:

    .venv/Scripts/python.exe profile_query.py
"""

import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "config.env")

import app as backend  # noqa: E402
import benchmark_qubo as bq  # noqa: E402
import providers  # noqa: E402

QUERY = "whats basepaper 11 about"


class Stage:
    """Context manager that records wall-clock time and an optional call count."""

    rows = []

    def __init__(self, name, calls=None):
        self.name, self.calls = name, calls

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        Stage.rows.append((self.name, time.perf_counter() - self.t0, self.calls))


async def main():
    user_id = await bq.resolve_user_id()
    total0 = time.perf_counter()

    with Stage("1. query rewriting", calls=1) as st:
        variants, _ = await backend.rewrite_queries(QUERY, num_variants=3)

    with Stage("2. vector retrieval", calls=0) as st:
        candidates = await backend.multi_query_retrieval(
            query_variants=variants, match_count_per_query=7, user_id=user_id
        )
    n_cand = len(candidates)

    with Stage(f"3. LLM reranking", calls=n_cand) as st:
        reranked, _ = await backend.rerank_chunks(QUERY, candidates, top_k=20)

    chunks = [c for c, _ in reranked] or candidates
    scores = [s for _, s in reranked] or [c.get("similarity", 0.0) for c in candidates]
    embeddings = [[0.0] * backend.EMBEDDING_DIM] * len(chunks)

    with Stage("4. QUBO annealing", calls=0) as st:
        selected, diag = backend.solve_chunk_selection(
            chunks=chunks, rf_scores=scores, embeddings=embeddings, token_budget=3000
        )

    context = "\n\n---\n\n".join(chunks[i]["content"] for i in selected)
    with Stage("5. answer generation", calls=1) as st:
        text, _ = await providers.llm_complete(
            f"User: {QUERY}\n\nAssistant:",
            max_tokens=700,
            temperature=0.3,
            system=f"Answer using ONLY this context.\n\n{context}",
            timeout=180.0,
        )

    total = time.perf_counter() - total0

    print("\n" + "=" * 64)
    print(f"LATENCY BREAKDOWN — {n_cand} candidates, concurrency "
          f"{providers.LLM_MAX_CONCURRENCY}")
    print("=" * 64)
    print(f"{'stage':<26}{'seconds':>10}{'share':>9}{'LLM calls':>12}")
    print("-" * 64)
    for name, secs, calls in Stage.rows:
        calls_s = "-" if not calls else str(calls)
        print(f"{name:<26}{secs:>10.2f}{secs / total * 100:>8.1f}%{calls_s:>12}")
    print("-" * 64)
    print(f"{'TOTAL':<26}{total:>10.2f}{'100.0%':>9}")
    print(f"\nanswer: {len((text or '').split())} words")


if __name__ == "__main__":
    asyncio.run(main())
