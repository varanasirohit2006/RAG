# Results review — 19 November 2026

The uploaded CSV and manifest are consistent with real, recorded pipeline output.
No fabricated replacement scores or aggregate mismatches were found. However,
this is a **partial, exploratory run**, not a completed accuracy benchmark. It
cannot establish that one answer-generation strategy is generally better.

## Evidence checked

- `summary.csv` is byte-identical to the saved run's summary.
- `manifest.json` matches the saved run's manifest.
- Every CSV aggregate, count, percentile and confidence interval reconciles with
  recomputation from the 24 distinct `(question, strategy)` sample records.
- The uploaded PDF's SHA-256 matches the declared document hash; it has 101 pages.
- All 10 supporting quotations for the six generated questions occur in their
  declared indexed chunks, and all reference page numbers are within the PDF.
- Completed metric values are finite; missing values remain unavailable rather
  than becoming artificial zeros.

This verifies provenance, source quotation matching and arithmetic consistency.
It does not independently calibrate the model's judgments or prove that synthetic
reference answers are complete human ground truth.

## Completion and failures

Run: `0d440ada9ea42eaa1e999c1c59688a2f4a99d68a8a6a39a7d552bc24eb3d3c8c`.
The same Gemini 3.1 Flash Lite model generated questions/answers and judged them.

| Outcome | Samples |
| --- | ---: |
| Fully completed | 4 |
| Partial | 4 |
| Failed | 2 |
| Skipped | 14 |
| Total | 24 |

Only **31 of 120** possible answer-quality metric results completed. No question
has a complete quality comparison across all four strategies. Means often use
only one or two samples, and the scored questions differ between strategies.
Ranking those quality means would confound the strategy with which questions
managed to finish.

The recorded failures include Google HTTP 503 generation errors, 90-second
individual judge-request timeouts, and a Google HTTP 429 rate/quota rejection.
The manifest does not identify the precise Google quota bucket or reset time.
Credits alone do not demonstrate sufficient request quota, and the provider
failure records do not prove that billing is absent or misconfigured.

The CSV's `failed` sample count includes partial attempts. For example, Keyword
has two partial samples and no completely failed samples, so its `error_rate`
is zero while its combined failed/partial rate is nonzero. These counts reconcile.

## What the retrieval results show

Retrieval was measured on all six questions for every strategy, even when later
answer generation or judging was blocked. These are the comparable results in
this particular run, using its synthetic relevance labels and cutoff of three.

| Strategy | Recall@3 | Precision@3 | nDCG@3 | MRR |
| --- | ---: | ---: | ---: | ---: |
| Keyword | 0.917 | 0.389 | 0.936 | 1.000 |
| Vector | 0.833 | 0.333 | 0.788 | 0.889 |
| Hybrid | 0.917 | 0.389 | 0.936 | 1.000 |
| Reranking | 1.000 | 0.444 | 0.987 | 1.000 |

Reranking matched Keyword on five questions and improved retrieval on one,
`auto-322fc0b7ca10921f`. That case's recall increased by 0.5, precision by 0.333,
and nDCG by approximately 0.307; MRR did not change. Reranking added approximately
0.633 seconds of reranking work per query in this run. This is evidence of one
useful retrieval improvement, not a statistically established general winner.

Faithfulness of 1.0 on a few answers does not mean 100% system accuracy. It measures
support in supplied context and can coexist with low reference-based factual
correctness. Same-model judging, incomplete relevance labels, only six synthetic
questions, and the large missing-score fraction further limit interpretation.
Latency means also include failed/cached/blocked attempts as documented; they
are not successful-response latency guarantees.

## A more defensible comparison

Use a completed run where all strategies score the same reviewed questions.
Keep actual source evidence and per-sample traces, review alternative relevant
passages, include unanswerable and diverse document cases, and compare strategies
on paired questions. For claims about accuracy, use human calibration and an
independent judge or independent human review. Keep the current partial run as
failure evidence; do not replace its missing scores with invented values.
