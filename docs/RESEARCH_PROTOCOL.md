# Research Protocol

## Title

**Better Router Adaptive: Quality-Cost-Latency-Aware Routing of Large Language Models Using Supervised Learning and Contextual Bandits**

## Research question

Can learned routing policies improve expected utility over a deterministic weighted routing policy when selecting among four heterogeneous language-model arms?

## Hypotheses

- **H1:** XGBoost achieves higher mean utility than the deterministic baseline on the held-out test set.
- **H2:** LinUCB reduces cumulative regret after its cold-start period while adapting to task-group differences.
- **H3:** At least one learned router improves the quality-cost frontier without increasing the error rate beyond the deterministic baseline.

## Methods locked before results

- Four model arms and four task groups.
- Prompt-level `70/15/15` train, validation, and test split.
- Seeds: `42`, `123`, `2026`, `31415`, `271828`.
- Algorithms: XGBoost and disjoint LinUCB.
- Baselines: deterministic Better Router proxy and offline oracle.
- Primary utility: `0.65*quality - 0.20*normalized_cost - 0.10*normalized_latency - 0.05*error`.
- Paired bootstrap confidence intervals resampled by prompt ID.
- No test-set tuning and no manual metric values in the report.

## Privacy and ethics

The academic phase uses public benchmark data and clearly marked synthetic fixtures. Production prompts, model responses, API credentials, user identifiers, and organization identifiers are excluded. Dataset licenses and provenance must be documented before experimental use.

## Reporting discipline

The repository distinguishes planned methods, generated evidence, and interpretation. No superiority claim is permitted before a reproducible run produces the corresponding result and confidence interval.
