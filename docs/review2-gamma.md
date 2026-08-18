# Q-RAG: A QUBO-Driven Efficient Information Retrieval System

Review 2 — Domain, Literature, Methodology and Design

**Team:** Aditi Babu · Kartik Iyer · Jefery Jose

19 August 2026

---

## The Domain

**Retrieval-Augmented Generation, viewed as a combinatorial optimization problem.**

Large language models cannot answer questions about documents they were never trained on. Retrieval-Augmented Generation (RAG) solves this by fetching relevant passages at question time and placing them in the model's prompt.

Our work sits at the intersection of three fields:

- **Information Retrieval** — finding passages relevant to a query
- **Combinatorial Optimization** — choosing the best subset under constraints
- **Quantum-Inspired Computing** — QUBO formulations and annealing solvers

The contribution is in how passages are *selected*, not how they are *found*.

---

## The Problem

**Standard RAG selects passages by ranking. Ranking cannot see redundancy.**

The universal implementation is *top-k by similarity*: score every passage against the question, sort, take the best k.

This has a structural blind spot. Similarity scores each passage **against the query** and never compares passages **to each other**.

If a document states a fact three times in different words, all three passages score highly and all three are retrieved. You have paid three times for one fact.

**Three consequences:**

1. **Cost** — context tokens are billed on every question, repeatedly
2. **Latency** — larger prompts mean slower responses
3. **Dilution** — every slot spent on a repeat is a slot denied to the passage that actually holds the answer

---

## Why This Matters Now

Context is the dominant recurring cost in deployed RAG systems.

- Prompt tokens are charged per query, forever — unlike a one-time indexing cost
- Context windows are finite; redundancy directly displaces useful content
- On edge and embedded deployments, prompt size translates into latency and power

**The core insight:** ranking asks *"how good is this passage?"* one at a time. The question that matters is *"how good is this **set** of passages?"* — and set quality depends on interactions between members, which no ranking function can express.

That makes it a combinatorial optimization problem.

---

## Literature Review — Retrieval

| Work | Contribution | Limitation for our problem |
|---|---|---|
| Robertson & Zaragoza (2009) — BM25 | Probabilistic lexical ranking; still a strong baseline | Term-matching only; scores passages independently |
| Karpukhin et al. (2020) — DPR | Dense passage retrieval via learned embeddings | Semantic, but still independent per-passage scoring |
| Lewis et al. (2020) — RAG | Established the retrieve-then-generate paradigm | Selection is top-k; redundancy unaddressed |
| Nogueira & Cho (2019) — BERT re-ranking | Cross-encoder reranking improves ordering | Improves *order*, not *set composition* |
| Gao et al. (2023) — RAG survey | Consolidates the modern RAG landscape | Identifies redundancy as open, does not solve it |

**Common thread:** every method above assigns each passage an independent score. None model the relationship *between* selected passages.

---

## Literature Review — Diversity and Optimization

| Work | Contribution | Limitation for our problem |
|---|---|---|
| Carbonell & Goldstein (1998) — MMR | First to trade relevance against redundancy | **Greedy**: commits to each pick permanently; no hard budget over variable-length passages |
| Glover, Kochenberger & Du (2019) | Tutorial on formulating problems as QUBO | General framework; not applied to retrieval |
| Lucas (2014) — Ising formulations | Maps many NP-hard problems to Ising/QUBO | Theoretical catalogue; retrieval not among them |
| Kirkpatrick et al. (1983) | Simulated annealing for global optimization | The solver we adopt |
| Kadowaki & Nishimori (1998) | Quantum annealing in the transverse Ising model | Establishes the hardware path our formulation targets |

> **Note for the paper:** verify all citation details (venue, page numbers, DOI) against the original sources before submission.

---

## Patent Landscape — Search Direction

For the patent review component, the relevant search space is defined by classification rather than by any single filing:

- **G06F 16/33** — information retrieval, query processing
- **G06F 16/93** — document management and retrieval
- **G06N 10/60** — quantum computing applications, optimization
- **G06N 5/04** — inference methods

**Search terms to run:** *context selection language model*, *prompt compression*, *retrieval diversity optimization*, *QUBO combinatorial selection*, *quantum annealing resource allocation*.

**Expected finding to verify:** prompt-compression and context-selection patents cluster around summarization and token pruning; formulating passage selection as a constrained binary optimization appears comparatively unexplored.

---

## Research Gap

Three gaps, in increasing order of importance:

1. **Independent scoring.** Retrieval methods score passages against the query alone. Pairwise redundancy between candidates is never represented.

2. **Greedy diversity.** MMR and its descendants handle diversity, but greedily — each selection is final and cannot be revised in light of later choices. There is also no principled way to enforce a hard token budget when passages have different lengths.

3. **No joint formulation.** Relevance, redundancy, and budget are optimized separately or heuristically, never as one objective with a well-defined optimum.

> **Our thesis:** passage selection is a *quadratic* problem, and treating it as one lets us state redundancy directly, enforce budgets exactly, and optimize the set jointly rather than greedily.

---

## Proposed Methodology — The QUBO

**QUBO** = Quadratic Unconstrained Binary Optimization. A set of yes/no decisions where cost depends on individual choices *and on pairs of choices*.

Let `xᵢ ∈ {0,1}` be 1 if passage *i* is selected. Minimize:

```
E(x) = − α · Σ sᵢxᵢ                    reward relevance
       + β · Σ tᵢxᵢ                    charge for tokens
       + γ · Σᵢ<ⱼ rᵢⱼxᵢxⱼ               penalize redundant pairs
       + κ · (Σxᵢ − k)²                 anchor selection size
       + P · (Σtᵢxᵢ + Σ2ᵇyᵇ − B)²       enforce token budget
```

| Symbol | Meaning |
|---|---|
| `sᵢ` | Relevance score (Random Forest + reranker), 0–1 |
| `tᵢ` | Passage tokens as a fraction of the budget |
| `rᵢⱼ` | Cosine similarity between passages *i* and *j* |
| `yᵇ` | Binary slack bits encoding unused budget |

Every term is at most quadratic in binary variables — a QUBO in the strict sense, mapping to an Ising model without reformulation.

---

## Why QUBO Specifically

**1. The quadratic term is the whole point.** `γ·rᵢⱼxᵢxⱼ` states *"these two passages are redundant with each other"* — the exact statement ranking cannot make.

**2. Joint, not greedy.** The optimizer evaluates whole sets and can revise any decision. MMR cannot undo an early commitment.

**3. Exact budget handling.** A ceiling `Σtᵢxᵢ ≤ B` is an inequality; QUBO admits only equality penalties. Slack variables `yᵇ` convert one into the other — the textbook construction, applied here to variable-length passages.

**4. Hardware portability.** The same matrix runs on CPU annealing today and on Ising machines, digital annealers, or quantum annealers without reformulation — a one-line sampler change.

**5. Anytime behaviour.** Annealing can be stopped at a deadline and return its best-so-far solution. Quality degrades gracefully with less compute; greedy algorithms offer no such control.

---

## System Design

**Five stages. Stages 1–3 are conventional and shared with the baseline, so the comparison isolates selection.**

**1 · Ingest** — Documents (PDF, DOCX, TXT, CSV, XLSX) parsed, split into ~1000-character passages, embedded into 384-dimensional vectors using a local ONNX model.

**2 · Retrieve** — Vector similarity search over the passage store returns a candidate pool.

**3 · Score** — Two independent signals blended 70/30: a trained **Random Forest** over 12 structural features (similarity, position, keyword overlap, length, numeric content), and an **LLM reranker** producing a semantic relevance judgement.

**4 · Select — the QUBO** ← *our contribution.* Build the cost matrix; minimize by simulated annealing.

**5 · Generate** — Selected passages are placed in a grounded prompt instructing the model to answer only from that context, and to say so when the answer is absent.

---

## Design Decisions Worth Defending

**Cardinality anchor (κ).** Without it, the mathematical optimum is to select almost nothing — token cost falls, a single passage cannot form a redundant pair, and *average* relevance rises because only the best passage remains. Every headline metric improves while the answer degrades. The anchor forces savings to come from choosing compact, non-overlapping passages rather than from abstaining.

**Adaptive redundancy floor.** The similarity threshold above which a pair counts as duplicated is calibrated per query from the candidate pool's own distribution, not fixed. A constant threshold is embedding-model-dependent and can silently mark *every* pair redundant, collapsing the redundancy term into a second cardinality penalty.

**Derived budget penalty (P).** A penalty enforces a constraint only if violating it can never pay for itself. `P` is computed from the other weights rather than hand-tuned, so the budget keeps binding when other parameters change.

---

## Evaluation Design

**Measuring this honestly requires care, because the obvious metrics are gameable.**

| Metric | Why it cannot be trusted alone |
|---|---|
| Token savings | Trivially maximized by retrieving almost nothing |
| Average relevance | It is a **mean** — keeping only the single best passage raises it while worsening the answer |

**Our evaluation protocol:**

- **Shared candidate pool.** All methods select from identical candidates under an identical budget, so differences are attributable to selection alone.
- **Blind LLM-as-judge.** Answers are labelled A/B/C with the mapping randomized per query, so the judge cannot favour any method.
- **Information coverage.** Fraction of *distinct* high-relevance clusters still represented. Dropping a duplicate is free; dropping unique content is penalized. This is the guard against a selector that merely abstains.
- **Falsifiability.** On problems small enough to enumerate exhaustively, the annealer's solution is compared against the brute-force optimum.

**Baselines:** top-k by similarity, and a greedy diversity heuristic.

---

## Applications in Robotics

**Strip away the documents and the problem is:** *choose a subset of items, under a hard resource budget, maximizing value while penalizing overlap.*

That pattern recurs throughout robotics, because robots are defined by hard budgets — finite compute, memory, bandwidth, battery, and a control loop with a deadline.

| Application | Mapping to our formulation |
|---|---|
| **On-robot LLM context** | A robot with an onboard language or vision-language planner must choose which observations, manual pages, and map regions enter the prompt under a hard token budget. This is our problem unchanged — only the corpus differs. |
| **SLAM keyframe selection** | Which keyframes remain in the pose graph under a memory budget? Overlapping views are redundant by definition. |
| **Active perception / next-best-view** | Choosing viewpoints or sensor streams to capture under bandwidth limits; two views of the same surface are redundant. |
| **Multi-robot task allocation** | The canonical QUBO application. Our budget term becomes robot capacity; our redundancy term becomes conflict penalty. |
| **Grasp and trajectory pruning** | Planners generate hundreds of near-identical candidates; a small, diverse, high-quality subset must be evaluated within a planning deadline. |

**The robotics-specific argument:** annealing is an *anytime* algorithm. Given a control-loop deadline — *"you have 8 milliseconds, return something"* — it returns its best solution so far. Greedy algorithms cannot be interrupted safely.

---

## Technology Stack

| Layer | Choice |
|---|---|
| Frontend | React + Vite |
| Backend | Python / FastAPI |
| Database | Supabase (PostgreSQL + pgvector), 384-dimensional vectors |
| Embeddings | BAAI/bge-small-en-v1.5, local ONNX inference |
| Language model | DeepSeek-V3 via Yotta / Shakti Cloud |
| Optimizer | dwave-neal + dimod (simulated annealing) |
| Token accounting | Native model tokenizer, not character estimation |

**Design principle:** embeddings run locally and the optimizer is classical, so the system has no dependency on paid inference for its core contribution.

---

## Future Plan

**Immediate — through Review 3**

1. **Complete the implementation** — finish the full evaluation harness and consolidate the comparison interface
2. **Full experimental evaluation** — expand the query set and adopt pairwise preference judging for statistical stability
3. **Parameter study** — systematic sweep over the objective weights (α, β, γ, κ) to characterize the relevance–cost frontier

**Toward publication**

4. **Complete the research paper** — finalize all chapters, formal results, and comparative analysis
5. **Complete the patent review** — execute the classification search and document findings

**Beyond**

6. Evaluate on larger, heterogeneous corpora where candidate pools are bigger and the optimization is harder
7. Port the formulation to annealing hardware — a sampler substitution, no reformulation required

---

## Summary

**Q-RAG reformulates context selection in RAG from a ranking problem into a constrained combinatorial optimization problem.**

- **Problem:** top-k retrieval cannot express redundancy between passages, so it pays repeatedly for the same information and dilutes the context window
- **Gap:** existing diversity methods are greedy and cannot enforce exact budgets over variable-length passages
- **Method:** a QUBO encoding relevance, token cost, pairwise redundancy, and a hard budget with slack variables — solved by simulated annealing
- **Design:** a five-stage pipeline in which only the selection stage differs from the baseline, making the comparison clean
- **Transfer:** the same formulation underlies keyframe selection, viewpoint planning, and multi-robot task allocation

**Team:** Aditi Babu · Kartik Iyer · Jefery Jose
