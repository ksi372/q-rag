"""
Honest benchmark: baseline RAG vs greedy heuristic vs QUBO annealing.

Why this exists
---------------
The in-app comparison panel reports token savings and `avg_rf_score`. Both are
easy to win dishonestly:

  * `avg_rf_score` is a MEAN. Keep only the single best chunk and it rises
    mechanically, regardless of whether the answer got better.
  * Token savings is trivially maximized by retrieving almost nothing.

So a selector that abstains looks superb on the dashboard and answers worse. This
script measures the thing those metrics proxy for — whether the generated answer
is actually good — and reports token cost alongside it. If QUBO only wins on
tokens while losing on answer quality, that shows up here instead of being hidden
behind a green verdict.

What it measures per query, per method
-------------------------------------
  chunks, tokens        context size actually sent to the model
  avg_score             mean RF/rerank score of the selection (the gameable one)
  redundant_pairs       selected pairs above the redundancy threshold
  coverage              fraction of the *baseline's* high-relevance chunks the
                        method retained. Reported but not the one to read: it
                        also punishes dropping near-duplicates, which is the
                        behaviour the redundancy term is meant to produce.
  info_coverage         fraction of *distinct* high-relevance clusters that are
                        still represented. This is the anti-abstention guard —
                        discarding a duplicate is free, discarding distinct
                        content is not, so a selector cannot look good by
                        retrieving less.
  answer_quality        blind LLM-judge score, 0-10, of the answer produced from
                        that method's context
  quality_per_1k        answer_quality per 1000 context tokens — the efficiency
                        number that actually means something

Judging is blind and order-randomized per query: the judge sees the answers
labelled A/B/C with the mapping shuffled, so it cannot prefer "the QUBO one".

Usage
-----
    .venv/Scripts/python.exe benchmark_qubo.py
    .venv/Scripts/python.exe benchmark_qubo.py --queries 8 --no-judge
    .venv/Scripts/python.exe benchmark_qubo.py --sweep
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / "config.env")

# The app defaults these OFF for demo latency, but the published comparison
# numbers were measured with the full pipeline. Force them on here so a
# benchmark run does not silently measure a different system than the one the
# results describe.
import os as _os

# The two optional LLM stages are set explicitly rather than inherited from
# config.env, so a benchmark run can never silently measure a different pipeline
# than the results claim. Default is the full pipeline; flags disable stages
# independently so their costs can be attributed separately.
_fast = "--fast" in sys.argv
_no_rewrite = _fast or "--no-rewrite" in sys.argv
_no_rerank = _fast or "--no-rerank" in sys.argv
_os.environ["ENABLE_QUERY_REWRITING"] = "false" if _no_rewrite else "true"
_os.environ["ENABLE_LLM_RERANKING"] = "false" if _no_rerank else "true"

import app as backend  # noqa: E402
import providers  # noqa: E402
import qubo  # noqa: E402
from qubo import QuboWeights  # noqa: E402

REDUNDANCY_THRESHOLD = 0.85
COVERAGE_SCORE_QUANTILE = 0.6  # "high-relevance" = top 40% of baseline scores

# Held fixed across methods so any difference comes from selection, not budget.
TOKEN_BUDGET = 3000
CANDIDATE_POOL = 20

DEFAULT_QUERIES = [
    "How does QUBO-based selection reduce redundancy in retrieved context?",
    "What retrieval approach does the architecture use and why?",
    "How are chunk relevance scores computed?",
    "What are the limitations of the proposed method?",
    "How does the system handle token budgets during retrieval?",
    "What datasets or corpora were used for evaluation?",
    "How does Q-RAG differ from a dense retrieval baseline?",
    "What role does the random forest play in the pipeline?",
]


# =============================================
# Retrieval (shared candidate pool)
# =============================================

async def build_candidate_pool(query: str, user_id: str) -> Optional[dict]:
    """
    Retrieve once and score once; every method then selects from the SAME pool.

    Sharing the pool is what makes the comparison about selection. If each method
    retrieved separately, differences in candidates would be confounded with
    differences in selection.
    """
    supabase = backend.get_supabase_client()

    variants, rewrite_tokens = await backend.rewrite_queries(
        query, num_variants=backend.ENHANCED_BASELINE_NUM_VARIANTS
    )
    candidates = await backend.multi_query_retrieval(
        query_variants=variants,
        match_count_per_query=max(CANDIDATE_POOL // max(len(variants), 1), 3),
        user_id=user_id,
    )
    if not candidates:
        return None

    reranked, rerank_tokens = await backend.rerank_chunks(
        query, candidates, top_k=backend.ENHANCED_BASELINE_RERANK_TOP_K
    )
    if not reranked:
        reranked = [(c, c.get("similarity", 0.0)) for c in candidates]

    chunks = [c for c, _ in reranked]
    rerank_scores = {c["id"]: s for c, s in reranked}

    doc_ids = list({c["document_id"] for c in chunks})
    docs = supabase.table("documents").select("id, filename").in_("id", doc_ids).execute()
    filenames = {d["id"]: d["filename"] for d in (docs.data or [])}

    chunk_counts = {}
    for doc_id in doc_ids:
        res = supabase.table("chunks").select("id", count="exact").eq("document_id", doc_id).execute()
        chunk_counts[doc_id] = res.count or 1

    embeddings, scores = [], []
    for chunk in chunks:
        detail = supabase.table("chunks").select("embedding").eq("id", chunk["id"]).execute()
        emb = None
        if detail.data and detail.data[0].get("embedding"):
            emb = backend.parse_embedding(detail.data[0]["embedding"])
        embeddings.append(emb or [0.0] * backend.EMBEDDING_DIM)

        features = backend.extract_features_for_scoring(
            query=query,
            chunk_content=chunk["content"],
            chunk_index=chunk["chunk_index"],
            total_chunks=chunk_counts.get(chunk["document_id"], 1),
            token_length=chunk["token_length"],
            similarity=chunk.get("similarity", 0.0),
        )
        rf = backend.score_chunk_with_rf(features)
        scores.append(
            backend.combine_rf_and_reranking_score(
                rf_score=rf,
                reranking_score=rerank_scores.get(chunk["id"], chunk.get("similarity", 0.0)),
                rf_weight=0.7,
                reranking_weight=0.3,
            )
        )

    return {
        "query": query,
        "chunks": chunks,
        "scores": scores,
        "embeddings": embeddings,
        "filenames": filenames,
        "redundancy": qubo.pairwise_cosine(embeddings),
        "aux_tokens": rewrite_tokens + rerank_tokens,
    }


# =============================================
# Selection strategies
# =============================================

def select_baseline(pool: dict, budget: int) -> List[int]:
    """Top-scoring chunks until the budget is spent — the standard approach."""
    order = sorted(range(len(pool["scores"])), key=lambda i: pool["scores"][i], reverse=True)
    selected, tokens = [], 0
    for i in order:
        t = int(pool["chunks"][i]["token_length"])
        if tokens + t > budget:
            continue
        selected.append(i)
        tokens += t
    return selected


def select_greedy(pool: dict, budget: int) -> List[int]:
    return backend.solve_qubo_greedy(
        chunks=pool["chunks"],
        rf_scores=pool["scores"],
        embeddings=pool["embeddings"],
        token_budget=budget,
        redundancy_threshold=REDUNDANCY_THRESHOLD,
        min_rf_score=0.05,
    )


def select_annealing(pool: dict, budget: int, weights: QuboWeights) -> Tuple[List[int], dict]:
    result = qubo.select_chunks(
        scores=pool["scores"],
        token_lengths=[int(c["token_length"]) for c in pool["chunks"]],
        embeddings=pool["embeddings"],
        token_budget=budget,
        weights=weights,
        solver="annealing",
        redundancy_matrix=pool["redundancy"],
    )
    return result.selected, {
        "energy": round(result.energy, 4),
        "variables": result.num_variables,
        "slack_bits": result.num_slack_bits,
        "budget_repairs": result.diagnostics.get("budget_repairs", 0),
        "redundancy_floor": result.diagnostics.get("redundancy_floor_used"),
    }


# =============================================
# Metrics
# =============================================

def context_for(pool: dict, selected: List[int]) -> str:
    parts = []
    for i in selected:
        chunk = pool["chunks"][i]
        name = pool["filenames"].get(chunk["document_id"], "Unknown")
        parts.append(f"[Source: {name}]\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def cluster_high_relevance(pool: dict, keep: set) -> List[set]:
    """
    Group the high-relevance chunks into near-duplicate clusters.

    Raw coverage ("what fraction of high-relevance chunks did you keep?") is the
    wrong guard on its own: it punishes dropping a near-duplicate, which is
    precisely the edit the redundancy term is supposed to make. Clustering fixes
    that — a cluster counts as covered if ANY of its members was selected, so
    discarding duplicates is free while discarding distinct information is not.

    Single-link agglomeration at the redundancy threshold; the corpus is small
    enough that transitive chaining isn't a practical concern.
    """
    red = pool["redundancy"]
    clusters: List[set] = []
    for i in sorted(keep):
        placed = False
        for cluster in clusters:
            if any(red[i, j] > REDUNDANCY_THRESHOLD for j in cluster):
                cluster.add(i)
                placed = True
                break
        if not placed:
            clusters.append({i})
    return clusters


def measure(pool: dict, selected: List[int], baseline_keep: set, clusters: List[set]) -> dict:
    scores = pool["scores"]
    red = pool["redundancy"]
    chosen = set(selected)

    tokens = providers.count_tokens(context_for(pool, selected))
    redundant = sum(
        1
        for a in range(len(selected))
        for b in range(a + 1, len(selected))
        if red[selected[a], selected[b]] > REDUNDANCY_THRESHOLD
    )
    raw_coverage = len(baseline_keep & chosen) / len(baseline_keep) if baseline_keep else 1.0
    info_coverage = (
        sum(1 for c in clusters if c & chosen) / len(clusters) if clusters else 1.0
    )
    return {
        "chunks": len(selected),
        "tokens": tokens,
        "avg_score": statistics.fmean([scores[i] for i in selected]) if selected else 0.0,
        "redundant_pairs": redundant,
        "coverage": raw_coverage,
        "info_coverage": info_coverage,
        "selected": selected,
    }


async def generate_answer(query: str, context: str) -> str:
    if not context:
        return "(no context selected)"
    system = (
        "Answer the question using ONLY the document context provided. "
        "If the context lacks the answer, say so plainly.\n\n"
        f"DOCUMENT CONTEXT:\n{context}"
    )
    try:
        text, _ = await providers.llm_complete(
            f"User: {query}\n\nAssistant:",
            max_tokens=400,
            temperature=0.2,
            system=system,
            timeout=120.0,
        )
        return text or "(empty response)"
    except Exception as e:
        return f"(generation failed: {e})"


async def judge_answers(query: str, answers: Dict[str, str], rng: random.Random) -> Dict[str, float]:
    """
    Blind, order-randomized LLM-as-judge scoring on 0-10.

    The judge never learns which method produced which answer, and the label
    order is shuffled per query so a positional bias can't systematically favour
    one method.
    """
    methods = list(answers.keys())
    rng.shuffle(methods)
    labels = {m: chr(ord("A") + i) for i, m in enumerate(methods)}

    blocks = "\n\n".join(
        f"--- Answer {labels[m]} ---\n{answers[m][:1500]}" for m in methods
    )
    prompt = (
        f"Question: {query}\n\n"
        f"{blocks}\n\n"
        "Score each answer 0-10 for how well it answers the question: factual "
        "support, completeness, and absence of padding or repetition. An answer "
        "that correctly states the documents lack the information should score "
        "well, not badly. Reply with ONLY a JSON object mapping each answer "
        f"letter to its score, e.g. {{\"A\": 7, \"B\": 4}}. Letters: "
        f"{', '.join(labels[m] for m in methods)}"
    )

    try:
        text, _ = await providers.llm_complete(
            prompt, max_tokens=200, temperature=0.0, timeout=120.0
        )
        start, end = text.find("{"), text.rfind("}")
        parsed = json.loads(text[start : end + 1]) if start >= 0 else {}
        return {
            m: float(parsed.get(labels[m], 0.0))
            for m in methods
            if labels[m] in parsed
        }
    except Exception as e:
        print(f"  [judge] failed: {e}")
        return {}


# =============================================
# Runner
# =============================================

async def run(queries: List[str], weights: QuboWeights, use_judge: bool, seed: int = 3) -> dict:
    user_id = await resolve_user_id()
    rng = random.Random(seed)
    rows: List[dict] = []

    for n, query in enumerate(queries, 1):
        print(f"\n[{n}/{len(queries)}] {query}")
        pool = await build_candidate_pool(query, user_id)
        if pool is None:
            print("  no candidates; skipped")
            continue

        # "High-relevance" set is defined by the baseline's own scoring, so
        # coverage isn't defined by the method being evaluated.
        scores = pool["scores"]
        threshold = float(np.quantile(scores, COVERAGE_SCORE_QUANTILE)) if scores else 0.0
        baseline_pick = select_baseline(pool, TOKEN_BUDGET)
        baseline_keep = {i for i in baseline_pick if scores[i] >= threshold}
        clusters = cluster_high_relevance(pool, baseline_keep)

        selections = {
            "baseline": select_baseline(pool, TOKEN_BUDGET),
            "greedy": select_greedy(pool, TOKEN_BUDGET),
        }
        annealed, diag = select_annealing(pool, TOKEN_BUDGET, weights)
        selections["annealing"] = annealed

        metrics = {
            m: measure(pool, sel, baseline_keep, clusters) for m, sel in selections.items()
        }
        print(f"  ({len(baseline_keep)} high-relevance chunks in {len(clusters)} distinct clusters)")
        for m, mm in metrics.items():
            print(
                f"  {m:<10} {mm['chunks']:>2} chunks  {mm['tokens']:>5} tok  "
                f"score {mm['avg_score']:.3f}  redundant {mm['redundant_pairs']:>2}  "
                f"info-cov {mm['info_coverage']:.2f}"
            )
        print(f"  annealing energy {diag['energy']}  ({diag['variables']} vars, "
              f"{diag['slack_bits']} slack, {diag['budget_repairs']} repairs, "
              f"redundancy floor {diag['redundancy_floor']})")

        if use_judge:
            answers = {}
            for m, sel in selections.items():
                answers[m] = await generate_answer(query, context_for(pool, sel))
            verdict = await judge_answers(query, answers, rng)
            for m in metrics:
                metrics[m]["answer_quality"] = verdict.get(m)
            if verdict:
                print("  judge: " + "  ".join(f"{m}={verdict.get(m)}" for m in metrics))

        rows.append({"query": query, "metrics": metrics, "diagnostics": diag})

    return {"rows": rows, "weights": weights.as_dict()}


async def resolve_user_id() -> str:
    """Owner of the corpus — whoever has documents."""
    if backend.USE_LOCAL_STORE:
        import local_store

        return local_store.LOCAL_USER_ID

    supabase = backend.get_supabase_client()
    res = supabase.table("documents").select("user_id").limit(1).execute()
    if not res.data:
        sys.exit("No documents in the store. Upload a corpus first.")
    return res.data[0]["user_id"]


def summarize(result: dict) -> None:
    rows = result["rows"]
    if not rows:
        print("\nNo results.")
        return

    methods = ["baseline", "greedy", "annealing"]
    fields = [
        "chunks", "tokens", "avg_score", "redundant_pairs",
        "coverage", "info_coverage", "answer_quality",
    ]

    print("\n" + "=" * 78)
    print(f"AGGREGATE over {len(rows)} queries")
    print("=" * 78)
    header = f"{'metric':<18}" + "".join(f"{m:>18}" for m in methods)
    print(header)
    print("-" * len(header))

    agg: Dict[str, Dict[str, Optional[float]]] = {}
    for field in fields:
        line = f"{field:<18}"
        agg[field] = {}
        for m in methods:
            vals = [
                r["metrics"][m][field]
                for r in rows
                if r["metrics"][m].get(field) is not None
            ]
            if not vals:
                agg[field][m] = None
                line += f"{'n/a':>18}"
                continue
            mean = statistics.fmean(vals)
            agg[field][m] = mean
            line += f"{mean:>18.3f}" if field != "tokens" else f"{mean:>18.0f}"
        print(line)

    # Quality per 1k tokens: the number that resists both gaming directions.
    print("-" * len(header))
    line = f"{'quality/1k tok':<18}"
    eff = {}
    for m in methods:
        q, t = agg.get("answer_quality", {}).get(m), agg["tokens"][m]
        if q is None or not t:
            eff[m] = None
            line += f"{'n/a':>18}"
        else:
            eff[m] = q / (t / 1000.0)
            line += f"{eff[m]:>18.3f}"
    print(line)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)

    base_t = agg["tokens"]["baseline"]
    ann_t = agg["tokens"]["annealing"]
    if base_t:
        print(f"tokens vs baseline      : {(base_t - ann_t) / base_t * 100:+.1f}%  "
              f"({base_t:.0f} -> {ann_t:.0f})")

    bq = agg.get("answer_quality", {}).get("baseline")
    aq = agg.get("answer_quality", {}).get("annealing")
    if bq and aq:
        print(f"answer quality vs base  : {aq - bq:+.2f} points  ({bq:.2f} -> {aq:.2f})")
        if eff["baseline"] and eff["annealing"]:
            ratio = eff["annealing"] / eff["baseline"]
            print(f"quality per token       : {ratio:.2f}x baseline")
        if aq >= bq - 0.5 and ann_t < base_t:
            print("\n=> QUBO WINS: fewer tokens at comparable-or-better answer quality.")
        elif ann_t < base_t:
            print(f"\n=> MIXED: saves {(base_t - ann_t) / base_t * 100:.0f}% tokens but answer "
                  f"quality drops {bq - aq:.2f} points. Not a clean win.")
        else:
            print("\n=> QUBO LOSES on tokens. See the tuning notes below.")
    else:
        print("(no judge scores — rerun without --no-judge for the quality verdict)")

    gq = agg.get("answer_quality", {}).get("greedy")
    if gq and aq:
        print(f"\nannealing vs old greedy : quality {aq - gq:+.2f}, "
              f"tokens {agg['tokens']['annealing'] - agg['tokens']['greedy']:+.0f}, "
              f"coverage {agg['coverage']['annealing'] - agg['coverage']['greedy']:+.2f}")

    print(f"\ncoverage (retained high-relevance baseline chunks):")
    for m in methods:
        print(f"  {m:<10} {agg['coverage'][m]:.2f}")
    print("  A method with low coverage is abstaining, not optimizing —")
    print("  cheap tokens there are not a real win.")


async def sweep(queries: List[str]) -> None:
    """
    Grid over the objective weights, ranked by quality per token.

    Tuning on measured answer quality rather than on avg_score is the whole
    point: optimizing avg_score directly rewards abstention.
    """
    grid = [
        ("relevance-heavy", QuboWeights(alpha=1.0, beta=0.25, gamma=1.2, kappa=0.30, k_target=6)),
        ("balanced (default)", QuboWeights()),
        ("token-thrifty", QuboWeights(alpha=1.0, beta=0.80, gamma=1.6, kappa=0.30, k_target=4)),
        ("redundancy-hostile", QuboWeights(alpha=1.0, beta=0.45, gamma=3.0, kappa=0.30, k_target=5)),
        ("compact (k=3)", QuboWeights(alpha=1.0, beta=0.60, gamma=2.0, kappa=0.50, k_target=3)),
    ]

    print("=" * 78)
    print(f"WEIGHT SWEEP — {len(grid)} configs x {len(queries)} queries")
    print("=" * 78)

    table = []
    for name, w in grid:
        print(f"\n### {name}: {w.as_dict()}")
        result = await run(queries, w, use_judge=True)
        rows = result["rows"]
        if not rows:
            continue
        tok = statistics.fmean([r["metrics"]["annealing"]["tokens"] for r in rows])
        cov = statistics.fmean([r["metrics"]["annealing"]["info_coverage"] for r in rows])
        qs = [
            r["metrics"]["annealing"]["answer_quality"]
            for r in rows
            if r["metrics"]["annealing"].get("answer_quality") is not None
        ]
        q = statistics.fmean(qs) if qs else None
        table.append((name, tok, cov, q, (q / (tok / 1000)) if q and tok else None))

    print("\n" + "=" * 78)
    print(f"{'config':<22}{'tokens':>10}{'coverage':>11}{'quality':>10}{'q/1k':>10}")
    print("-" * 78)
    for name, tok, cov, q, eff in sorted(
        table, key=lambda r: (r[4] is None, -(r[4] or 0))
    ):
        qs = f"{q:.2f}" if q is not None else "n/a"
        es = f"{eff:.3f}" if eff is not None else "n/a"
        print(f"{name:<22}{tok:>10.0f}{cov:>11.2f}{qs:>10}{es:>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=5, help="how many queries to run")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM answer scoring")
    parser.add_argument("--sweep", action="store_true", help="grid over objective weights")
    parser.add_argument("--out", help="write raw results to this JSON path")
    parser.add_argument("--fast", action="store_true",
                        help="disable both optional LLM stages")
    parser.add_argument("--no-rewrite", action="store_true",
                        help="disable query rewriting only")
    parser.add_argument("--no-rerank", action="store_true",
                        help="disable LLM reranking only")
    args = parser.parse_args()

    queries = DEFAULT_QUERIES[: max(1, args.queries)]

    print(f"store    : {'local-sqlite' if backend.USE_LOCAL_STORE else 'supabase'}")
    print(f"llm      : {providers.llm_model_name()} (available={providers.llm_available()})")
    print(f"embedding: {providers.provider_status()['embedding_model']} "
          f"({backend.EMBEDDING_DIM}d)")
    print(f"rf model : {'loaded' if backend.RF_MODEL is not None else 'MISSING'}")
    print(f"budget   : {TOKEN_BUDGET} tokens, pool {CANDIDATE_POOL}")
    print(f"pipeline : rewriting={backend.ENABLE_QUERY_REWRITING} reranking={backend.ENABLE_LLM_RERANKING}")

    if args.sweep:
        asyncio.run(sweep(queries))
        return

    result = asyncio.run(run(queries, backend.QUBO_OBJECTIVE_WEIGHTS, not args.no_judge))
    summarize(result)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"\nraw results -> {args.out}")


if __name__ == "__main__":
    main()
