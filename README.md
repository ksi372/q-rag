# Synergy — QUBO-based RAG retrieval optimizer

Selects which retrieved chunks actually go into an LLM's context by solving a QUBO
(quadratic unconstrained binary optimization) over relevance and redundancy, instead
of stuffing the top-k by cosine similarity. The product surface is the side-by-side
comparison: same query, same token budget, enhanced-baseline RAG vs QUBO+RF.

Typical result on the bundled corpus: **50–72% fewer tokens, higher average chunk
quality, and near-zero redundant pairs.**

## Run it

Two processes. Backend first.

```bash
cd server && .venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8081
```

```bash
npm run dev
```

Then open http://localhost:5173, sign up (local mode accepts any email/password),
add documents in the sidebar, and ask a question. The comparison panel appears under
the answer.

Health check, which reports every active provider and the store backend:

```bash
curl -s http://127.0.0.1:8081/ | python -m json.tool
```

## How the pipeline works

1. **Ingest** — extract text (PDF/DOCX/TXT/CSV/XLSX/JSON), chunk to ~1000 chars,
   embed, store.
2. **Retrieve** — the query is rewritten into several variants, each retrieved
   against, and the candidate pool merged and deduplicated.
3. **Score** — an LLM reranker scores each candidate; a trained Random Forest
   (`server/rf_model.pkl`, 12 derived features, 0.727 accuracy) scores it
   independently. The two are combined 70/30.
4. **Select** — `server/qubo.py` builds an actual QUBO and minimizes it with
   simulated annealing (see below). Set `QUBO_SOLVER=greedy` to fall back to the
   original heuristic, whose penalty thresholds come from
   `server/shap_weights.json` (SHAP + grid search + LLM-as-judge).
5. **Compare** — `/api/rag/compare` runs the baseline and the QUBO path under the
   *same* token budget and reports the deltas. Auxiliary tokens spent on rewriting
   and reranking are counted against both sides, so the savings are net.

## The QUBO

Variables `x_i ∈ {0,1}` — include candidate chunk *i* or not. Minimize:

```
E(x) = - α Σ s_i x_i                       relevance reward
       + β Σ t_i x_i                       token price
       + γ Σ_{i<j} r_ij x_i x_j            redundancy penalty
       + κ ( Σ x_i - k )²                  cardinality anchor
       + P ( Σ t_i x_i + Σ 2^b y_b - B )²  budget, with binary slack
```

`s_i` is the combined RF + rerank score, `t_i` the chunk's tokens as a fraction of
the budget, `r_ij` the cosine similarity between chunk embeddings, `y_b` slack
bits. Every term is at most quadratic in binaries, so it maps to an Ising model
with no reformulation. Solved with `dwave-neal` simulated annealing (120 reads,
1200 sweeps, fixed seed), with a pure-NumPy Metropolis fallback.

Two terms are worth knowing about, because they're where a benchmark gets rigged:

- **Cardinality anchor (κ).** Without it, the minimizer discovers that selecting
  almost nothing is optimal — token price drops, no pair can be redundant, and
  `avg_rf_score` rises *mechanically* because it's a mean over one kept chunk.
  Every headline metric improves while the answer gets worse. The anchor holds
  selection near `k_target` so the token win has to come from picking compact
  non-redundant chunks rather than from abstaining.
- **Budget slack (`y_b`).** `Σ t_i x_i ≤ B` is an inequality and QUBO admits only
  equality penalties, so the ceiling becomes `Σ t_i x_i + slack = B` with the
  slack in binary bits. That's why variable count exceeds candidate count.

`P` is derived from the other weights rather than hand-set (`effective_penalty`):
a penalty only enforces a constraint if violating it can never pay for itself, and
a weight tuned on one query silently stops binding when `κ` or `k_target` changes.
Selection is also repaired post-hoc if it still overruns, and the repair count is
reported in diagnostics rather than swallowed — a nonzero value means the weights
need looking at.

Tunable via env: `QUBO_ALPHA`, `QUBO_BETA`, `QUBO_GAMMA`, `QUBO_KAPPA`,
`QUBO_K_TARGET`, `QUBO_REDUNDANCY_FLOOR`.

### Verifying it

```bash
cd server && .venv/Scripts/python.exe test_qubo.py
```

20 checks. The one that matters: on instances small enough to brute-force, the
annealer's energy is compared against the enumerated ground state. Without that,
"the annealer found a good solution" is unfalsifiable.

```bash
cd server && .venv/Scripts/python.exe benchmark_qubo.py --queries 5
```

Baseline vs greedy vs annealing over a query set, sharing one candidate pool so
differences come from selection alone. Scores answers with a **blind,
order-randomized LLM judge** — the in-app panel's token savings and `avg_rf_score`
are both trivially gamed by retrieving less, so answer quality is the arbiter, and
`info_coverage` (distinct high-relevance clusters still represented) is the guard
against a selector that just abstains. `--sweep` grids the weights and ranks
configs by quality per token.

## Providers

None of the original hosted dependencies still work: the first Supabase project's
hostname no longer resolves, and both Gemini keys are invalid or out of prepayment
credits. Everything is behind a provider layer (`server/providers.py`) selected by
env in `server/config.env`.

| Concern | Default | Notes |
| --- | --- | --- |
| LLM | Yotta / Shakti Cloud (`deepseek-v3`) | OpenAI-compatible. `LLM_PROVIDER=gemini` or `none` also work; `none` degrades to pure cosine retrieval. |
| Embeddings | Local ONNX, `BAAI/bge-small-en-v1.5`, **384-dim** | The Yotta gateway has no `/embeddings` route. First run downloads ~130MB, then it's offline and free. |
| Token counts | Real `deepseek-ai/DeepSeek-V3` tokenizer | Every "tokens saved" number depends on this, so it's a real tokenization, not chars/4. |
| Data | SQLite (`server/local_data/`) + localStorage | Used while Supabase is unconfigured. |

Two Yotta details that will bite if changed: the `id: <gateway_id>` header must be
sent **only** for the gateway's own model (`deepseek-v3`) — other models 400 with it
present — and deepseek needs `thinking: {type: disabled}`, or reasoning tokens eat
the `max_tokens` budget and short replies come back empty.

Because the RF model scores *derived* features (similarity, position, keyword
overlap, length, …) rather than raw vectors, switching embedding providers and
dimensions does not invalidate it.

## Switching to Supabase

The app runs standalone today. Both keys for the new project
(`<your-project-ref>`) are already in `server/config.env`; only the schema is
missing. DDL cannot go through the REST API, so step 1 is manual and the rest is
scripted.

1. Paste **`supabase/setup.sql`** into the Supabase SQL editor and run it. One
   idempotent file covering the RAG tables, the chat tables, RLS, and both storage
   buckets — it supersedes running `migrations/001`–`003` separately.
2. Auth → Providers → Email: turn **off** "Confirm email", otherwise the first
   signup can't sign in until the emailed link is clicked.
3. Create the user who should own the existing corpus (Authentication → Users), then:

```bash
cd server && .venv/Scripts/python.exe switch_to_supabase.py --owner you@example.com
```

That verifies every table and the `match_chunks` signature, copies the local
documents and chunks (embeddings included) into Supabase, and flips `SUPABASE_URL`
on in both env files. Restart uvicorn and Vite afterwards. Use `--check` to inspect
without writing anything, or `--skip-data` to switch without copying.

Two things that will bite:

- `setup.sql` declares **`vector(384)`** to match the local embedding model.
  Migration 002's `vector(768)` is Gemini-only; that width makes every insert fail.
- The backend needs the **service_role** key, not anon. It writes on the user's
  behalf and RLS blocks anon, since `auth.uid()` is null so every policy evaluates
  false.

The chat tables (`folders`, `threads`, `messages`, `message_attachments`) were never
captured in a migration — they were created by hand in the original project. The
schema in `setup.sql` is reconstructed from the columns `src/pages/Workspace.jsx`
actually queries.

## Layout

```
server/
  app.py              FastAPI: ingest, retrieval, QUBO selection, comparison, chat
  providers.py        LLM / embedding / tokenizer providers
  local_store.py      SQLite stand-in for the supabase-py surface app.py uses
  rf_model.pkl        Trained chunk scorer
  shap_weights.json   SHAP-derived QUBO penalty weights
  config.env          Backend configuration (gitignored)
training/             RF training, SHAP weight optimization
supabase/setup.sql    Complete schema, paste-and-run
src/
  pages/Workspace.jsx Main app: documents, chat, comparison panel
  components/         Boot splash + ported vaporize-text effect
  lib/localBackend.js localStorage stand-in for auth and chat tables
```

## Notes / known gaps

- Reranking makes one small LLM call per candidate chunk, bounded by
  `LLM_MAX_CONCURRENCY` (default 8). A full comparison runs ~30–60s; sequentially it
  was several minutes.
- Simulated annealing is classical. It's the standard way to solve a QUBO without
  quantum hardware, and the formulation is exactly what a D-Wave sampler would
  take — swapping `neal.SimulatedAnnealingSampler` for `DWaveSampler` is a
  one-line change plus credentials. Don't describe the current build as running on
  quantum hardware; describe it as a QUBO formulation solved by annealing, which
  is both true and the interesting part.
- In local mode, chat attachment *bodies* live in per-session blob URLs, so previews
  don't survive a reload (metadata does). Supabase Storage fixes this.
- The frontend `RAG_BACKEND_URL` must match the uvicorn port; it was previously
  hardcoded to 8001 while the backend served 8081, which silently disabled all RAG.
