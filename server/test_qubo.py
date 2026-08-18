"""
Correctness checks for the QUBO formulation.

The point is falsifiability: if the annealer's answer can't be checked against a
known ground state, "the annealer works" is an assertion. These compare against
brute-force enumeration on problems small enough to enumerate, and verify each
objective term actually does what its name claims.

Run: .venv/Scripts/python.exe test_qubo.py
"""

import numpy as np

import qubo
from qubo import QuboWeights, build_qubo, energy_of, pairwise_cosine, select_chunks

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))
    print(f"[{PASS if condition else FAIL}] {name}" + (f"  — {detail}" if detail else ""))


def orthogonal_embeddings(n: int, dim: int = 16) -> list:
    """Distinct unit vectors, so redundancy between them is ~0."""
    M = np.zeros((n, dim))
    for i in range(n):
        M[i, i % dim] = 1.0
    return M.tolist()


# ---------------------------------------------------------------------------
# 1. Energy bookkeeping: the matrix must reproduce the objective by hand.
# ---------------------------------------------------------------------------

def test_energy_matches_manual():
    scores = [1.0, 0.0]
    tokens = [100, 100]
    red = np.zeros((2, 2))
    w = QuboWeights(alpha=1.0, beta=0.0, gamma=0.0, kappa=0.0, penalty=0.0)
    Q, n, slack = build_qubo(scores, tokens, red, 1000, w)

    # Normalized scores become [1, 0]; only the relevance term survives.
    e_none = energy_of(Q, np.array([0.0, 0.0] + [0.0] * slack))
    e_first = energy_of(Q, np.array([1.0, 0.0] + [0.0] * slack))
    e_both = energy_of(Q, np.array([1.0, 1.0] + [0.0] * slack))

    check("energy of empty selection is 0", abs(e_none) < 1e-9, f"E={e_none:.6f}")
    check("selecting the top chunk pays -alpha", abs(e_first + 1.0) < 1e-9, f"E={e_first:.6f}")
    check("zero-score chunk is free at beta=0", abs(e_both + 1.0) < 1e-9, f"E={e_both:.6f}")


# ---------------------------------------------------------------------------
# 2. Redundancy term: a near-duplicate must be rejected.
# ---------------------------------------------------------------------------

def test_redundancy_rejects_duplicate():
    # Three chunks; #0 and #1 are near-identical, #2 is unrelated.
    emb = [[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]]
    red = pairwise_cosine(emb)
    check(
        "duplicate pair reads as highly similar",
        red[0, 1] > 0.95,
        f"r01={red[0,1]:.4f}, r02={red[0,2]:.4f}",
    )

    scores = [0.9, 0.9, 0.6]
    tokens = [100, 100, 100]
    w = QuboWeights(alpha=1.0, beta=0.1, gamma=2.0, kappa=0.0, penalty=0.0, k_target=2)
    res = select_chunks(scores, tokens, emb, 1000, w, solver="exact", redundancy_matrix=red)

    got_both_dupes = 0 in res.selected and 1 in res.selected
    check(
        "ground state does not take both near-duplicates",
        not got_both_dupes,
        f"selected={res.selected} solver={res.solver}",
    )


# ---------------------------------------------------------------------------
# 3. Token price: given equal scores, prefer the cheaper chunk.
# ---------------------------------------------------------------------------

def test_token_price_prefers_cheap():
    emb = orthogonal_embeddings(2)
    scores = [0.8, 0.8]
    tokens = [100, 600]
    w = QuboWeights(alpha=1.0, beta=1.0, gamma=0.0, kappa=0.0, penalty=0.0, k_target=1)
    res = select_chunks(scores, tokens, emb, 1000, w, solver="exact")
    # Equal raw scores normalize to 0.5 each, so only token price separates them.
    check(
        "cheaper chunk preferred at equal relevance",
        res.selected == [0] or (0 in res.selected and 1 not in res.selected),
        f"selected={res.selected} tokens={res.tokens_used}",
    )


# ---------------------------------------------------------------------------
# 4. Cardinality anchor: without it the optimum collapses to ~nothing.
#    This is the term that keeps the benchmark honest, so it gets a direct test.
# ---------------------------------------------------------------------------

def test_cardinality_anchor_prevents_collapse():
    n = 8
    emb = orthogonal_embeddings(n)
    scores = [0.9 - 0.05 * i for i in range(n)]
    tokens = [200] * n

    # The collapse only threatens when the token price outweighs relevance — that
    # is the regime the anchor exists for, so test it there. At beta < alpha the
    # optimum is naturally large and the anchor *shrinks* it instead, which is a
    # different (also correct) behaviour and would make this assertion meaningless.
    #
    # tokens/budget = 200/2000 = 0.1, so the price per chunk is beta * 0.1. Scores
    # normalize onto [0,1], so beta = 12 puts the price (1.2) above even a perfect
    # score and nothing is worth taking on relevance alone.
    beta_high = 12.0
    no_anchor = QuboWeights(alpha=1.0, beta=beta_high, gamma=1.0, kappa=0.0)
    with_anchor = QuboWeights(alpha=1.0, beta=beta_high, gamma=1.0, kappa=0.5, k_target=4)

    a = select_chunks(scores, tokens, emb, 2000, no_anchor, solver="annealing")
    b = select_chunks(scores, tokens, emb, 2000, with_anchor, solver="annealing")

    check(
        "without the anchor a high token price collapses selection",
        len(a.selected) == 0,
        f"no_anchor selected {len(a.selected)} chunks",
    )
    check(
        "anchor restores a substantive selection",
        len(b.selected) > len(a.selected),
        f"no_anchor={len(a.selected)} chunks, anchor(k=4)={len(b.selected)} chunks",
    )
    check(
        "anchored selection lands near k_target",
        abs(len(b.selected) - 4) <= 1,
        f"selected {len(b.selected)}, target 4",
    )


# ---------------------------------------------------------------------------
# 5. Budget: the constraint must actually bind.
# ---------------------------------------------------------------------------

def test_budget_binds():
    n = 10
    emb = orthogonal_embeddings(n)
    scores = [0.9] * n
    tokens = [300] * n           # 3000 total available
    budget = 900                 # room for ~3

    # k_target deliberately exceeds what the budget can hold, so the anchor and
    # the budget constraint are in direct conflict. The budget must win.
    w = QuboWeights(alpha=1.0, beta=0.3, gamma=0.0, kappa=0.2, k_target=8)
    res = select_chunks(scores, tokens, emb, budget, w, solver="annealing")
    check(
        "selection respects the token budget",
        res.tokens_used <= budget,
        f"used {res.tokens_used} of {budget} ({len(res.selected)} chunks)",
    )
    check(
        "budget overrides an over-ambitious k_target",
        len(res.selected) < 8,
        f"k_target=8 but budget allowed {len(res.selected)}",
    )
    check(
        "derived penalty binds without post-hoc repair",
        res.diagnostics.get("budget_repairs", 0) == 0,
        f"repairs={res.diagnostics.get('budget_repairs')}, P={w.effective_penalty():.2f}",
    )


def test_repair_enforces_budget_when_penalty_too_low():
    """A deliberately weak penalty must still not leak an over-budget selection."""
    n = 6
    emb = orthogonal_embeddings(n)
    scores = [0.9] * n
    tokens = [300] * n
    budget = 600

    weak = QuboWeights(alpha=1.0, beta=0.0, gamma=0.0, kappa=0.5, penalty=0.001, k_target=6)
    res = select_chunks(scores, tokens, emb, budget, weak, solver="annealing")
    check(
        "repair enforces the ceiling even with a near-zero penalty",
        res.tokens_used <= budget,
        f"used {res.tokens_used} of {budget}, repairs={res.diagnostics.get('budget_repairs')}",
    )
    check(
        "repair is reported rather than hidden",
        res.diagnostics.get("budget_repairs", 0) > 0,
        f"repairs={res.diagnostics.get('budget_repairs')}",
    )


# ---------------------------------------------------------------------------
# 6. Annealer quality: does it actually find the ground state?
# ---------------------------------------------------------------------------

def test_annealing_matches_exact():
    rng = np.random.default_rng(11)
    matches, gaps, trials = 0, [], 12

    for trial in range(trials):
        n = 9
        emb = rng.normal(size=(n, 12)).tolist()
        scores = rng.uniform(0.2, 1.0, size=n).tolist()
        tokens = rng.integers(80, 400, size=n).tolist()
        w = QuboWeights(k_target=4)

        red = pairwise_cosine(emb)
        Q, nc, slack = build_qubo(scores, tokens, red, 1500, w)

        ex = qubo.solve_exact(Q, nc, tokens, 1500, max_variables=24)
        if ex is None:
            continue
        an = qubo.solve_annealing(Q, nc, tokens, 1500)

        gap = an.energy - ex.energy
        gaps.append(gap)
        if abs(gap) < 1e-6:
            matches += 1

    worst = max(gaps) if gaps else 0.0
    check(
        "annealer reaches the exact ground state on small instances",
        matches == len(gaps),
        f"{matches}/{len(gaps)} exact, worst energy gap {worst:.2e}",
    )


# ---------------------------------------------------------------------------
# 7. Determinism: same input, same context.
# ---------------------------------------------------------------------------

def test_deterministic():
    rng = np.random.default_rng(5)
    n = 14
    emb = rng.normal(size=(n, 16)).tolist()
    scores = rng.uniform(0.2, 1.0, size=n).tolist()
    tokens = rng.integers(100, 400, size=n).tolist()
    w = QuboWeights(k_target=5)

    runs = [select_chunks(scores, tokens, emb, 2500, w, solver="annealing").selected for _ in range(3)]
    check(
        "repeated solves return identical selections",
        all(r == runs[0] for r in runs),
        f"selections={[len(r) for r in runs]}",
    )


# ---------------------------------------------------------------------------
# 8. Degenerate inputs must not raise.
# ---------------------------------------------------------------------------

def test_edge_cases():
    w = QuboWeights()
    empty = select_chunks([], [], [], 1000, w)
    check("empty candidate set returns empty selection", empty.selected == [], f"{empty.selected}")

    single = select_chunks([0.7], [120], [[1.0, 0.0]], 1000, w, solver="annealing")
    check("single candidate is handled", isinstance(single.selected, list), f"{single.selected}")

    ragged = select_chunks(
        [0.5, 0.6], [100, 100], [[1.0, 0.0], []], 1000, w, solver="annealing"
    )
    check("missing embedding does not crash", isinstance(ragged.selected, list), f"{ragged.selected}")

    zero_budget = select_chunks([0.9, 0.8], [100, 100], orthogonal_embeddings(2), 0, w)
    check("zero budget is survivable", isinstance(zero_budget.selected, list), f"{zero_budget.selected}")


if __name__ == "__main__":
    print("=" * 74)
    print("QUBO formulation tests")
    print("=" * 74)
    for fn in [
        test_energy_matches_manual,
        test_redundancy_rejects_duplicate,
        test_token_price_prefers_cheap,
        test_cardinality_anchor_prevents_collapse,
        test_budget_binds,
        test_repair_enforces_budget_when_penalty_too_low,
        test_annealing_matches_exact,
        test_deterministic,
        test_edge_cases,
    ]:
        print(f"\n--- {fn.__name__} ---")
        fn()

    failed = [r for r in results if r[0] == FAIL]
    print("\n" + "=" * 74)
    print(f"{len(results) - len(failed)}/{len(results)} passed")
    if failed:
        for _, name, detail in failed:
            print(f"  FAIL: {name} — {detail}")
        raise SystemExit(1)
