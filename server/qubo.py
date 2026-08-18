"""
QUBO formulation and solvers for chunk selection.

The original `solve_qubo_greedy` was a greedy heuristic that *referenced* a QUBO
objective without ever building or minimizing one. This module writes the actual
Hamiltonian and hands it to a sampler, so "QUBO" is a claim that survives being
asked how it works.

--------------------------------------------------------------------------------
The objective
--------------------------------------------------------------------------------
Binary decision variables x_i in {0,1} — include candidate chunk i or not.

    E(x) = - alpha * SUM_i  s_i * x_i                     (relevance reward)
           + beta  * SUM_i  t_i * x_i                     (token price)
           + gamma * SUM_i<j r_ij * x_i * x_j             (redundancy penalty)
           + kappa * ( SUM_i x_i - k_target )^2           (cardinality anchor)
           + P     * ( SUM_i t_i x_i + SUM_b 2^b y_b - B )^2   (budget, with slack)

  s_i  in [0,1]  combined RF + reranking score
  t_i  in [0,1]  chunk tokens as a fraction of the budget
  r_ij in [0,1]  cosine similarity between chunk embeddings
  y_b            binary slack bits encoding budget headroom
  B    in [0,1]  budget, normalized to 1

Everything is at most quadratic in binaries, so this is a QUBO in the strict
sense and maps to an Ising model without reformulation.

Two terms deserve justification, because they are where a benchmark can be
quietly rigged:

*Cardinality anchor (kappa).* Without it the minimizer discovers that selecting
almost nothing is optimal: token price drops, no pair can be redundant, and
`avg_rf_score` — being a *mean* — rises simply because you kept only the single
best chunk. Every headline metric improves while the answer gets worse. The
anchor holds selection near k_target so the token win has to come from choosing
compact non-redundant chunks, not from abstaining. This is the honest-comparison
term; removing it inflates the win.

*Budget slack (y_b).* A ceiling of the form SUM t_i x_i <= B is an inequality,
and QUBO admits only equality penalties. The textbook fix is a slack variable
turning it into SUM t_i x_i + slack = B. The slack is encoded in
ceil(log2) binary bits, which is why the variable count exceeds the
candidate count.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# =============================================
# Weights
# =============================================

@dataclass
class QuboWeights:
    """
    Objective coefficients.

    Defaults are deliberately conservative: relevance dominates, the token price
    is real but not overwhelming, and redundancy is punished hardest, since
    dropping a near-duplicate is the one edit that is almost free in answer
    quality and pure profit in tokens.
    """

    alpha: float = 1.0      # relevance reward
    beta: float = 0.45      # token price
    gamma: float = 1.6      # redundancy penalty
    kappa: float = 0.30     # cardinality anchor
    # Budget weight. None = derive it from the other terms (see effective_penalty);
    # 0.0 = genuinely disable the budget term. Keeping these distinct matters:
    # collapsing them leaves no way to express "no budget constraint".
    penalty: Optional[float] = None
    k_target: int = 5       # anchor point for how many chunks to keep

    # Redundancy below this is treated as 0. None = calibrate per query from the
    # candidate pool's own similarity distribution (see redundancy_floor_quantile).
    #
    # A FIXED floor is a trap, and it bit this build: bge-small embeddings occupy
    # a narrow cone, so on a topically homogeneous corpus the *minimum* observed
    # pairwise similarity was 0.552 — above the 0.55 that was configured. Every
    # pair was therefore "redundant", which turns this term into a pure
    # cardinality penalty and collapses selection to 2-3 chunks regardless of
    # k_target. Calibrating makes "redundant" mean "more similar than this pool's
    # typical pair" rather than "above a constant that depends on which embedding
    # model you happen to be using".
    redundancy_floor: Optional[float] = None
    redundancy_floor_quantile: float = 0.80

    def effective_penalty(self) -> float:
        """
        Budget penalty weight, derived from the other terms unless set explicitly.

        A constraint encoded as a penalty is only respected if violating it can
        never pay for itself. The most any single chunk can earn is alpha (a
        perfect relevance score) plus the anchor's pull toward k_target, which is
        at most 2*kappa*k_target. Setting the penalty comfortably above that sum
        makes the budget effectively hard instead of a suggestion.

        Picking this by hand is how budget overruns get shipped: a weight tuned
        on one query silently stops binding when k_target or kappa changes.
        """
        if self.penalty is not None:
            return float(self.penalty)
        max_gain = self.alpha + 2.0 * self.kappa * max(self.k_target, 1) + self.gamma
        return 8.0 * max(max_gain, 1e-6)

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "kappa": self.kappa,
            "penalty": self.penalty,
            "penalty_effective": round(self.effective_penalty(), 4),
            "k_target": self.k_target,
            "redundancy_floor": self.redundancy_floor,
            "redundancy_floor_quantile": self.redundancy_floor_quantile,
        }


@dataclass
class QuboResult:
    selected: List[int]
    energy: float
    solver: str
    num_variables: int
    num_candidates: int
    num_slack_bits: int
    tokens_used: int
    diagnostics: dict = field(default_factory=dict)


# =============================================
# Matrix construction
# =============================================

def resolve_redundancy_floor(redundancy: Optional[np.ndarray], weights: QuboWeights) -> float:
    """
    The similarity above which a pair counts as redundant.

    Explicit setting wins. Otherwise it is the given quantile of the pool's own
    off-diagonal similarities, so the term penalizes pairs that are unusually
    similar *for this corpus and this embedding model* instead of comparing
    against an absolute constant. With fewer than two pairs there is nothing to
    calibrate against, so nothing is penalized.
    """
    if weights.redundancy_floor is not None:
        return float(weights.redundancy_floor)

    if redundancy is None or redundancy.size == 0 or redundancy.shape[0] < 3:
        return 1.0  # nothing to compare; disable the term rather than guess

    off = redundancy[np.triu_indices(redundancy.shape[0], 1)]
    off = off[off > 0]
    if off.size < 2:
        return 1.0
    return float(np.quantile(off, weights.redundancy_floor_quantile))


def build_qubo(
    scores: Sequence[float],
    token_lengths: Sequence[int],
    redundancy: np.ndarray,
    token_budget: int,
    weights: QuboWeights,
) -> Tuple[np.ndarray, int, int]:
    """
    Build the upper-triangular QUBO matrix Q where E(x) = x^T Q x.

    Returns (Q, n_candidates, n_slack_bits). Variables are ordered candidates
    first, then slack bits.
    """
    n = len(scores)
    if n == 0:
        return np.zeros((0, 0)), 0, 0

    budget = max(int(token_budget), 1)

    s = np.asarray(scores, dtype=np.float64)
    # Scores arrive on varying scales depending on whether reranking ran; map to
    # [0,1] so alpha means the same thing across queries.
    if s.max() > s.min():
        s = (s - s.min()) / (s.max() - s.min())
    else:
        s = np.ones_like(s) * 0.5

    t = np.asarray(token_lengths, dtype=np.float64) / budget  # normalized, B = 1

    # Slack bits span the budget in normalized units. Resolution finer than ~2%
    # of the budget buys nothing and just widens the search space.
    n_slack = max(1, min(8, int(math.ceil(math.log2(max(budget, 2))) - 4)))
    slack_scale = 1.0 / (2 ** n_slack - 1)

    total = n + n_slack
    Q = np.zeros((total, total), dtype=np.float64)

    # ---- linear: relevance reward, token price ----
    for i in range(n):
        Q[i, i] += -weights.alpha * s[i] + weights.beta * t[i]

    # ---- quadratic: redundancy ----
    floor = resolve_redundancy_floor(redundancy, weights)
    if redundancy is not None and redundancy.size and floor < 1.0:
        span = max(1.0 - floor, 1e-6)
        for i in range(n):
            for j in range(i + 1, n):
                r = float(redundancy[i, j])
                if r <= floor:
                    continue
                # Rescale so the floor maps to 0 and 1.0 stays 1.0: the penalty
                # should grow with excess similarity, not with absolute similarity.
                Q[i, j] += weights.gamma * (r - floor) / span

    # ---- cardinality anchor: kappa * (SUM x_i - k)^2 ----
    # Expands to kappa*(1 - 2k) on each diagonal and 2*kappa on each pair.
    k = float(weights.k_target)
    if weights.kappa > 0:
        for i in range(n):
            Q[i, i] += weights.kappa * (1.0 - 2.0 * k)
        for i in range(n):
            for j in range(i + 1, n):
                Q[i, j] += 2.0 * weights.kappa
        # The +kappa*k^2 constant is dropped: it shifts energy, not the argmin.

    # ---- budget: P * (SUM t_i x_i + slack - 1)^2 ----
    # Coefficient vector over all variables, then expand the square.
    P = weights.effective_penalty()
    if P > 0:
        c = np.zeros(total, dtype=np.float64)
        c[:n] = t
        for b in range(n_slack):
            c[n + b] = (2 ** b) * slack_scale

        for i in range(total):
            # c_i^2 x_i^2 - 2*c_i*x_i  (x^2 = x for binaries)
            Q[i, i] += P * (c[i] * c[i] - 2.0 * c[i])
        for i in range(total):
            for j in range(i + 1, total):
                Q[i, j] += P * 2.0 * c[i] * c[j]
        # The +P*1 constant is dropped for the same reason as above.

    return Q, n, n_slack


def energy_of(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x @ Q @ x)


# =============================================
# Solvers
# =============================================

def _decode(
    x: np.ndarray,
    n_candidates: int,
    token_lengths: Sequence[int],
    token_budget: int,
    scores: Optional[Sequence[float]] = None,
) -> Tuple[List[int], int, int]:
    """
    Read candidate bits out of a sample, then enforce the budget.

    The budget is encoded as a penalty, and a penalty is soft by construction: a
    sampler is always free to return an assignment that overspends if the rest of
    the objective pays for it. `effective_penalty` makes that unlikely, but
    "unlikely" is not a contract, and downstream code treats the token budget as
    a hard ceiling. So repair here: drop the worst score-per-token chunks until
    the selection fits.

    Returns (selected, tokens, num_repaired) — the repair count is surfaced in
    diagnostics rather than swallowed, because a nonzero value means the penalty
    weight is too low for the current weights and should be looked at.
    """
    selected = [i for i in range(n_candidates) if x[i] > 0.5]
    tokens = sum(int(token_lengths[i]) for i in selected)

    budget = max(int(token_budget), 0)
    if tokens <= budget or not selected:
        return selected, tokens, 0

    # Evict by worst value density first; ties broken by larger token cost.
    def density(i: int) -> tuple:
        t = max(int(token_lengths[i]), 1)
        s = float(scores[i]) if scores is not None and i < len(scores) else 0.0
        return (s / t, -t)

    order = sorted(selected, key=density)  # worst value first
    repaired = 0
    for i in order:
        if tokens <= budget:
            break
        selected.remove(i)
        tokens -= int(token_lengths[i])
        repaired += 1

    return selected, tokens, repaired


def solve_exact(
    Q: np.ndarray,
    n_candidates: int,
    token_lengths: Sequence[int],
    token_budget: int,
    max_variables: int = 22,
    scores: Optional[Sequence[float]] = None,
) -> Optional[QuboResult]:
    """
    Brute-force ground state. Only viable for small problems, but it gives a
    reference the heuristics can be scored against — without it, "the annealer
    found a good solution" is unfalsifiable.
    """
    total = Q.shape[0]
    if total == 0 or total > max_variables:
        return None

    best_x, best_e = None, math.inf
    for mask in range(1 << total):
        x = np.fromiter(((mask >> b) & 1 for b in range(total)), dtype=np.float64, count=total)
        e = energy_of(Q, x)
        if e < best_e:
            best_e, best_x = e, x

    selected, tokens, repaired = _decode(
        best_x, n_candidates, token_lengths, token_budget, scores
    )
    return QuboResult(
        selected=selected,
        energy=best_e,
        solver="exact",
        num_variables=total,
        num_candidates=n_candidates,
        num_slack_bits=total - n_candidates,
        tokens_used=tokens,
        diagnostics={"enumerated": 1 << total, "budget_repairs": repaired},
    )


def solve_annealing(
    Q: np.ndarray,
    n_candidates: int,
    token_lengths: Sequence[int],
    token_budget: int,
    num_reads: int = 120,
    num_sweeps: int = 1200,
    seed: Optional[int] = 7,
    scores: Optional[Sequence[float]] = None,
) -> QuboResult:
    """
    Simulated annealing over the QUBO via dwave-neal, with a NumPy fallback.

    A fixed seed by default: retrieval that returns different context for the
    same query is not debuggable, and a benchmark whose numbers move between
    runs cannot support a claim.
    """
    total = Q.shape[0]
    if total == 0:
        return QuboResult([], 0.0, "annealing", 0, 0, 0, 0)

    try:
        import neal

        # dimod wants {(i,j): bias}; our Q is already upper-triangular.
        qubo_dict: Dict[Tuple[int, int], float] = {}
        for i in range(total):
            for j in range(i, total):
                if Q[i, j] != 0.0:
                    qubo_dict[(i, j)] = float(Q[i, j])

        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample_qubo(
            qubo_dict, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed
        )
        best = sampleset.first
        x = np.array([best.sample.get(i, 0) for i in range(total)], dtype=np.float64)

        # Recompute from Q rather than trusting the reported energy: dimod drops
        # our discarded constants, and we want one consistent scale everywhere.
        energy = energy_of(Q, x)
        selected, tokens, repaired = _decode(
            x, n_candidates, token_lengths, token_budget, scores
        )

        energies = sampleset.record.energy
        return QuboResult(
            selected=selected,
            energy=energy,
            solver="annealing(neal)",
            num_variables=total,
            num_candidates=n_candidates,
            num_slack_bits=total - n_candidates,
            tokens_used=tokens,
            diagnostics={
                "num_reads": num_reads,
                "num_sweeps": num_sweeps,
                "distinct_energies": int(len(set(np.round(energies, 6)))),
                "energy_spread": float(energies.max() - energies.min()) if len(energies) else 0.0,
                "budget_repairs": repaired,
            },
        )
    except ImportError:
        return _solve_annealing_numpy(
            Q, n_candidates, token_lengths, token_budget, num_reads, num_sweeps, seed, scores
        )


def _solve_annealing_numpy(
    Q: np.ndarray,
    n_candidates: int,
    token_lengths: Sequence[int],
    token_budget: int,
    num_reads: int,
    num_sweeps: int,
    seed: Optional[int],
    scores: Optional[Sequence[float]] = None,
) -> QuboResult:
    """
    Metropolis single-flip simulated annealing, so the annealer path does not
    depend on dwave-neal being installable.
    """
    total = Q.shape[0]
    rng = np.random.default_rng(seed)

    # Symmetrize once; delta-energy for a single flip is then a dot product.
    S = Q + Q.T - np.diag(np.diag(Q))

    best_x, best_e = None, math.inf
    for _ in range(num_reads):
        x = (rng.random(total) < 0.5).astype(np.float64)
        e = energy_of(Q, x)
        t_hot, t_cold = 2.0, 0.01

        for sweep in range(num_sweeps):
            temp = t_hot * (t_cold / t_hot) ** (sweep / max(num_sweeps - 1, 1))
            for i in rng.permutation(total):
                # Flipping x_i from v to 1-v changes E by (1-2v)*(Q_ii + S_i.x - S_ii*x_i)
                v = x[i]
                delta = (1.0 - 2.0 * v) * (Q[i, i] + S[i] @ x - S[i, i] * v)
                if delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9)):
                    x[i] = 1.0 - v
                    e += delta

        e = energy_of(Q, x)  # guard against drift from incremental updates
        if e < best_e:
            best_e, best_x = e, x.copy()

    selected, tokens, repaired = _decode(
        best_x, n_candidates, token_lengths, token_budget, scores
    )
    return QuboResult(
        selected=selected,
        energy=best_e,
        solver="annealing(numpy)",
        num_variables=total,
        num_candidates=n_candidates,
        num_slack_bits=total - n_candidates,
        tokens_used=tokens,
        diagnostics={
            "num_reads": num_reads,
            "num_sweeps": num_sweeps,
            "budget_repairs": repaired,
        },
    )


# =============================================
# Entry point
# =============================================

def select_chunks(
    scores: Sequence[float],
    token_lengths: Sequence[int],
    embeddings: Sequence[Sequence[float]],
    token_budget: int,
    weights: Optional[QuboWeights] = None,
    solver: str = "annealing",
    redundancy_matrix: Optional[np.ndarray] = None,
) -> QuboResult:
    """
    Build the QUBO and minimize it.

    solver: "annealing" (default), "exact" (small problems only, falls back to
    annealing), or "auto" (exact when it fits, otherwise annealing).
    """
    weights = weights or QuboWeights()
    n = len(scores)
    if n == 0:
        return QuboResult([], 0.0, solver, 0, 0, 0, 0)

    if redundancy_matrix is None:
        redundancy_matrix = pairwise_cosine(embeddings)

    Q, n_candidates, n_slack = build_qubo(
        scores, token_lengths, redundancy_matrix, token_budget, weights
    )
    resolved_floor = resolve_redundancy_floor(redundancy_matrix, weights)

    result = None
    if solver in ("exact", "auto"):
        result = solve_exact(Q, n_candidates, token_lengths, token_budget, scores=scores)
        if result is None and solver == "exact":
            print("[QUBO] Problem too large for exact solve; using annealing")

    if result is None:
        result = solve_annealing(Q, n_candidates, token_lengths, token_budget, scores=scores)

    # Surface the floor that was actually applied: when it is calibrated per query
    # this is the single most useful number for explaining why the selection is
    # the size it is.
    result.diagnostics["redundancy_floor_used"] = round(resolved_floor, 4)
    return result


def pairwise_cosine(embeddings: Sequence[Sequence[float]]) -> np.ndarray:
    """Cosine similarity matrix, zero diagonal, robust to ragged/empty rows."""
    n = len(embeddings)
    if n == 0:
        return np.zeros((0, 0))

    widths = {len(e) for e in embeddings if e is not None}
    if not widths:
        return np.zeros((n, n))
    dim = max(widths)

    M = np.zeros((n, dim), dtype=np.float64)
    for i, e in enumerate(embeddings):
        if e is None:
            continue
        v = np.asarray(e, dtype=np.float64).ravel()
        if v.size == dim:
            M[i] = v
        elif v.size:
            # Mismatched width means a provider change; leave the row zero rather
            # than compare vectors from different embedding spaces.
            continue

    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    N = M / norms
    R = N @ N.T
    np.fill_diagonal(R, 0.0)
    return np.clip(R, 0.0, 1.0)
