# Research Roadmap

## Phase 1 — Reproducible foundation

- Python package and immutable experiment configuration.
- Unit tests, linting, typing, coverage, CI, citation metadata, and research protocol.

## Phase 2 — Dataset and provenance

- Acquire the official RouterBench data or an explicitly documented alternative.
- Record source, license, retrieval date, size, and SHA-256.
- Commit only compact synthetic fixtures and manifests.

## Phase 3 — Data pipeline

- Clean and normalize the benchmark into one row per `(prompt_id, model_id)`.
- Split by prompt ID to avoid leakage.
- Generate pre-inference features only.

## Phase 4 — Routing algorithms

- Deterministic Better Router proxy.
- Offline oracle.
- XGBoost utility predictor.
- Disjoint LinUCB contextual bandit.

## Phase 5 — Evaluation

- Mean quality, cost, latency, error rate, utility, top-1 oracle match, and regret.
- Paired bootstrap intervals and sensitivity analysis.
- Generated tables and figures tied to a run manifest.

## Phase 6 — Publication

- Final IEEE report.
- Scientific poster.
- Reproducibility package.
- Better AI research article.
- Optional archived release and DOI.

Production integration into Better Router is outside the academic phase and requires a separate design, shadow-mode evaluation, and deployment safety review.
