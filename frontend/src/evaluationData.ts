export const profiles = {
  full: { label: 'Full comparison', description: 'Test every question across keyword, vector, hybrid and reranking.', strategies: 4 },
  preview: { label: 'Quality preview', description: 'Test a smaller selected subset through the production pipeline.', strategies: 1 },
  retrieval: { label: 'Retrieval only', description: 'Compare keyword, vector and hybrid evidence selection. No RAG answers or judging.', strategies: 3 },
} as const
export type Profile = keyof typeof profiles
export const qualityMetrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'factual_correctness']
export const retrievalMetrics = ['precision_at_k', 'recall_at_k', 'mrr', 'ndcg_at_k', 'hit_rate_at_k', 'map_at_k']
export const labels: Record<string, string> = {
  faithfulness: 'Faithfulness', answer_relevancy: 'Answer relevancy', context_precision: 'Context precision', context_recall: 'Context recall', factual_correctness: 'Factual correctness',
  precision_at_k: 'Precision@k', recall_at_k: 'Recall@k', mrr: 'MRR', ndcg_at_k: 'nDCG@k', hit_rate_at_k: 'Hit rate@k', map_at_k: 'MAP@k',
  citation_page_validity: 'Citation page validity', citation_page_coverage: 'Citation page coverage', citation_source_validity: 'Citation source validity', normalized_exact_match: 'Exact match', token_f1: 'Token F1',
  multimodal_faithfulness: 'Visual faithfulness', abstention_accuracy: 'Abstention accuracy', false_answer_rate: 'False answer rate',
  samples: 'Samples', completed: 'Completed', failed: 'Failed', skipped: 'Skipped', failure_rate: 'Failure rate', partial_rate: 'Partial rate', error_rate: 'Error rate',
  sample_seconds_p50: 'Total p50 (s)', sample_seconds_p95: 'Total p95 (s)', retrieval_seconds_p50: 'Retrieval p50 (s)', retrieval_seconds_p95: 'Retrieval p95 (s)',
  generation_seconds_p50: 'Answer p50 (s)', generation_seconds_p95: 'Answer p95 (s)', judge_seconds_p50: 'Judge p50 (s)', judge_seconds_p95: 'Judge p95 (s)',
}
export const metricDescriptions: Record<string, string> = {
  faithfulness: 'Are the answer’s claims supported by the evidence sent to the generator?',
  answer_relevancy: 'How closely does the answer address the test question?',
  context_precision: 'How much of the retrieved evidence is useful for answering the question?',
  context_recall: 'Does the retrieved evidence cover the information in the reference answer?',
  factual_correctness: 'How well do the answer’s factual claims match the source-verified reference?',
}
export type MetricDefinition = { label?: string; description?: string; range?: unknown; direction?: string; method?: string; applicability?: unknown }
export function display(value: unknown): string {
  if (value == null || typeof value === 'number' && !Number.isFinite(value)) return '—'
  return typeof value === 'number' ? Number.isInteger(value) ? String(value) : value.toFixed(3) : String(value)
}
export function metricLabel(name: string, catalog: Record<string, MetricDefinition> = {}) {
  return catalog[name]?.label || labels[name] || name.replaceAll('_', ' ')
}
