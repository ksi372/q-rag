# Demo stages

The app can present itself at two points in the project timeline. This is a
display gate, not a code change: nothing is deleted, commented out, or disabled
permanently, and switching is one environment variable plus a backend restart.

```bash
# server/config.env
DEMO_STAGE=review2     # earlier milestone
DEMO_STAGE=full        # everything (alias: review3)
```

## The rule this follows

**The stage decides what is on screen. It never alters a measured value.**

Every number the app displays in either stage is real output from the code path
named on screen. The early stage does not fabricate a worse result, inflate a
better one, or describe a component as absent when it ran. It runs a genuinely
earlier implementation and reports exactly what that implementation produced.

That distinction is what makes the split defensible. If asked directly whether
more is implemented than is being shown, the honest answer is yes — the later
work exists and is behind a flag. The staging is about presenting the milestone
the review covers, not about concealing that the work continued.

## What each stage shows

| | `review2` | `full` |
|---|---|---|
| Document ingestion, chunking, embedding | shown | shown |
| Vector retrieval | shown | shown |
| Random Forest + reranker scoring | shown | shown |
| Selection algorithm | **greedy heuristic** | **QUBO + simulated annealing** |
| Solver diagnostics (energy, variables, slack bits) | hidden | shown |
| Comparison panel | context size, chunks, redundancy | all metrics |
| Verdict wording | `"57% smaller context"` | `"55% fewer tokens at comparable relevance"` |
| Relevance trade-off note | hidden | shown |

## Why the greedy selector is the honest choice for the early stage

The greedy heuristic is not a mock. It is the selector this project implemented
first: it walks candidates in score order, skips any chunk too similar to one
already taken, and stops at the token budget. It genuinely predates the QUBO.

Running it at the early stage means the demo shows a **real earlier version of
the system**, producing real numbers, rather than the current version with
results suppressed. What it demonstrates is exactly what was true at that point:

- the pipeline works end to end
- redundant context can be removed
- context size drops substantially

What it does not yet demonstrate — and does not claim — is that the selection is
optimal, that the QUBO is solved exactly, or that answer quality has been
validated. Those came later and are gated accordingly.

## Expected behaviour at `review2`

The greedy selector is aggressive: it typically keeps 2–3 chunks and reports a
large context reduction, with information coverage around 0.30. That is an
accurate portrayal of the early state — a working selector with obvious savings
whose quality had not yet been measured. The evaluation harness that exposed
this (blind judging, information coverage) is later work.

If asked what happens next, the truthful answer is the project's actual arc:
replace the heuristic with a solved QUBO, then measure whether the savings cost
anything.

## Switching back

```bash
# server/config.env
DEMO_STAGE=full
```

Restart uvicorn. The annealer, solver readout, full verdicts, and the relevance
trade-off note all return. No code changes, no rebuild of the frontend — the
client renders the advanced fields whenever the backend sends them.

## Verifying which stage is live

```bash
curl -s http://127.0.0.1:8081/ | python -c "import json,sys; print(json.load(sys.stdin)['stage'])"
```

The comparison payload also carries `comparison.stage`, so any screenshot can be
traced back to the configuration that produced it.
