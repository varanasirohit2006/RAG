import { useEffect, useRef, useState } from 'react'
import useSWR from 'swr'
import type { Config, Dataset, DocumentInfo, JobInfo, Run } from '../types'
import { isActive } from '../types'
import { post, uploadFile, useJob } from '../api'
import { ErrorMessage, JobProgress } from '../components'

const profiles = {
  full: ['Full comparison', 'Every test question across keyword, vector, hybrid and reranking. All four core quality metrics plus factual correctness.'],
  preview: ['Quality preview', 'Up to five test questions through the production pipeline. All four core quality metrics plus factual correctness.'],
  retrieval: ['Retrieval only', 'Keyword, vector and hybrid evidence selection, without generating RAG answers or judging them. Automatic question generation still uses API credits.'],
} as const
const qualityMetrics = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'factual_correctness']
const labels: Record<string, string> = {
  faithfulness: 'Faithfulness', answer_relevancy: 'Answer relevancy', context_precision: 'Context precision', context_recall: 'Context recall', factual_correctness: 'Factual correctness',
  precision_at_k: 'Precision@k', recall_at_k: 'Recall@k', mrr: 'MRR', ndcg_at_k: 'nDCG@k', hit_rate_at_k: 'Hit rate@k', map_at_k: 'MAP@k',
  citation_page_validity: 'Citation page validity', citation_page_coverage: 'Citation page coverage', citation_source_validity: 'Citation source validity', normalized_exact_match: 'Exact match', token_f1: 'Token F1',
  multimodal_faithfulness: 'Visual faithfulness', abstention_accuracy: 'Abstention accuracy', false_answer_rate: 'False answer rate',
}
type MetricDefinition = { label?: string; description?: string; range?: unknown; direction?: string; method?: string; applicability?: unknown }
type SummaryRow = Run['summary'][number]
function display(value: unknown): string {
  if (value === null || value === undefined) return '—'
  return typeof value === 'number' ? Number.isInteger(value) ? String(value) : value.toFixed(3) : String(value)
}
function metricLabel(name: string, catalog: Record<string, MetricDefinition>) { return catalog[name]?.label || labels[name] || name.replaceAll('_', ' ') }
function MetricTable({ rows, metrics, catalog, caption }: { rows: SummaryRow[]; metrics: string[]; catalog: Record<string, MetricDefinition>; caption: string }) {
  return <div className="table-scroll"><table className="metric-table"><caption>{caption}</caption><thead><tr><th scope="col">Strategy</th>{metrics.map(name => <th scope="col" key={name} title={catalog[name]?.description}>{metricLabel(name, catalog)}</th>)}</tr></thead>
    <tbody>{rows.map((row, index) => <tr key={`${row.strategy}-${index}`}><th scope="row">{display(row.strategy)}</th>{metrics.map(name => {
      const count = row[`${name}_count`]
      const failed = Number(row[`${name}_failed`] || 0), skipped = Number(row[`${name}_skipped`] || 0)
      const lower = row[`${name}_lower95`], upper = row[`${name}_upper95`]
      const interval = typeof lower === 'number' && typeof upper === 'number' ? `${display(lower)}–${display(upper)}` : null
      return <td key={name}><strong>{display(row[name])}</strong><small>{typeof count === 'number' ? `${count} valid / ${display(row.samples)}` : row[name] == null ? 'Unavailable' : 'Valid count unavailable'}</small>{failed || skipped ? <small className={failed ? 'metric-failure' : ''}>{failed} failed · {skipped} skipped / N/A</small> : null}{interval ? <small title="Bootstrap 95% confidence interval">95% CI {interval}</small> : null}</td>
    })}</tr>)}</tbody></table></div>
}
function QuestionCases({ dataset, documentId }: { dataset: Dataset; documentId: string }) {
  const synthetic = dataset.origin === 'synthetic' || dataset.provenance?.origin === 'synthetic'
  return <details className="dataset-details"><summary>Inspect {dataset.count} {synthetic ? 'generated' : 'saved'} questions and references</summary>
    <p className="muted">{synthetic ? 'Synthetic questions and reference answers checked automatically against source evidence. They have not been reviewed by a person.' : 'Imported cases are validated for document identity and evidence labels. Inspect their source references when interpreting results.'}</p>
    {dataset.warnings?.map((warning, index) => <div className="notice" key={index}>{warning}</div>)}
    {dataset.cases.map(c => <div className="review-case" key={c.id}><div className="row spread"><strong>{c.id}: {c.question}</strong>{c.category ? <span className="badge">{c.category.replaceAll('_', ' ')}</span> : null}</div><p>{c.reference_answer}</p>
      <div className="row wrap"><small>{c.answerable ? 'Source evidence:' : 'Unanswerable case'}</small>{c.reference_pages?.map(page => <a key={page} href={`/api/documents/${documentId}/pdf#page=${page}`} target="_blank" rel="noreferrer">Page {page} ↗</a>)}{c.relevant_chunk_ids?.length ? <small>{c.relevant_chunk_ids.length} labelled passage{c.relevant_chunk_ids.length === 1 ? '' : 's'}</small> : null}</div>
      {c.supporting_quotes?.map((quote, index) => <blockquote key={index}>{typeof quote === 'string' ? quote : quote.quote || quote.text || ''}</blockquote>)}
      {c.verification ? <details><summary>Source verification</summary><pre>{JSON.stringify(c.verification, null, 2)}</pre></details> : null}
    </div>)}
    {dataset.provenance ? <details><summary>Dataset provenance</summary><pre>{JSON.stringify(dataset.provenance, null, 2)}</pre></details> : null}
  </details>
}
function RunCard({ run }: { run: Run }) {
  const m = run.manifest
  const catalog = (m.metric_catalog && typeof m.metric_catalog === 'object' && !Array.isArray(m.metric_catalog) ? m.metric_catalog : {}) as Record<string, MetricDefinition>
  const columns = Array.from(new Set(run.summary.flatMap(row => Object.keys(row))))
  const additionalMetrics = columns.filter(name => !qualityMetrics.includes(name) && columns.includes(`${name}_count`) && !name.endsWith('_seconds'))
  const hasQuality = qualityMetrics.some(name => columns.includes(name))
  const operational = ['samples', 'completed', 'failed', 'skipped', 'failure_rate', 'partial_rate', 'error_rate', 'sample_seconds_p50', 'sample_seconds_p95', 'retrieval_seconds_p50', 'retrieval_seconds_p95', 'generation_seconds_p50', 'generation_seconds_p95', 'judge_seconds_p50', 'judge_seconds_p95'].filter(name => columns.includes(name))
  const warnings = [...new Set([...(Array.isArray(m.warnings) ? m.warnings : []), ...(Array.isArray(m.dataset_warnings) ? m.dataset_warnings : [])].map(String))]
  return <article className="run-card"><div className="row spread"><h3>{profiles[m.profile as keyof typeof profiles]?.[0] || display(m.profile)} evaluation</h3><span className={`badge ${m.status === 'completed' ? 'ready' : ''}`}>{display(m.status)}</span></div>
    <p className="muted">{display(m.sampled_questions)} / {display(m.total_questions)} questions · {display(m.completed_samples)} completed · {display(m.failed_samples)} failed or partial · {display(m.skipped_samples)} skipped · {display(m.seconds)}s</p>
    {run.cache_reused ? <div className="notice">Completed evaluation reused from cache.</div> : null}
    {warnings.map((warning, index) => <div className="notice" key={index}>{warning}</div>)}
    {m.error ? <ErrorMessage error={String(m.error)}/> : null}
    {run.summary.length ? <>{hasQuality ? <><h4>Answer quality comparison</h4><MetricTable rows={run.summary} metrics={qualityMetrics} catalog={catalog} caption="Mean valid scores by retrieval strategy. Missing scores are shown as unavailable."/></> : <p className="muted">This run has no answer quality scores. Retrieval-only runs measure evidence selection.</p>}
      {additionalMetrics.length ? <details open={!hasQuality}><summary>Retrieval, citations and additional quality metrics</summary><MetricTable rows={run.summary} metrics={additionalMetrics} catalog={catalog} caption="Additional metrics; validity counts and confidence intervals use their own applicable samples."/><p className="muted">Citation page checks measure citation structure. Lexical overlap measures wording. Neither proves semantic support or factual correctness.</p></details> : null}
      {operational.length ? <details><summary>Latency and reliability</summary><div className="table-scroll"><table><caption>Latency is measured in seconds; rates are fractions of samples.</caption><thead><tr><th scope="col">Strategy</th>{operational.map(name => <th scope="col" key={name}>{name.replaceAll('_', ' ')}</th>)}</tr></thead><tbody>{run.summary.map((row,index) => <tr key={index}><th scope="row">{display(row.strategy)}</th>{operational.map(name => <td key={name}>{display(row[name])}</td>)}</tr>)}</tbody></table></div><p className="muted">Percentiles describe this run only. Cached stage timings differ from fresh model execution.</p></details> : null}
    </> : <p>No completed metric summary. Inspect the run manifest for details.</p>}
    <div className="row exports">{(['summary','samples','manifest'] as const).map(kind => <a className="secondary" key={kind} href={`/api/runs/${run.id}/export/${kind}`}>Download {kind}</a>)}</div>
    {Object.keys(catalog).length ? <details><summary>Metric definitions & applicability</summary><div className="metric-definitions">{Object.entries(catalog).map(([name, definition]) => <div key={name}><strong>{metricLabel(name, catalog)}</strong><p>{definition.description}</p><small>{definition.method ? `Method: ${definition.method}. ` : ''}{definition.range ? `Range: ${JSON.stringify(definition.range)}. ` : ''}{definition.direction ? `Direction: ${definition.direction}. ` : ''}{definition.applicability ? `Applies to: ${typeof definition.applicability === 'string' ? definition.applicability : JSON.stringify(definition.applicability)}` : ''}</small></div>)}</div></details> : null}
    <details><summary>Configuration & methodology</summary><pre>{JSON.stringify(m,null,2)}</pre></details>
  </article>
}

export default function EvaluationView({ doc, config }: { doc: DocumentInfo; config: Config }) {
  const { data: datasets = [], error: datasetError, mutate: refreshDatasets } = useSWR<Dataset[]>(`/api/documents/${doc.id}/datasets`)
  const { data: runs = [], error: runsError, mutate: refreshRuns } = useSWR<Run[]>(`/api/documents/${doc.id}/runs`)
  const work = useJob<Run>(`evaluation:${doc.id}`)
  const [mode, setMode] = useState<'automatic' | 'saved'>('automatic')
  const [datasetId, setDatasetId] = useState('')
  const [questionCount, setQuestionCount] = useState('10')
  const [profile, setProfile] = useState<keyof typeof profiles>('full')
  const [force, setForce] = useState(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const dataset = datasets.find(d => d.id === datasetId) || datasets[0]
  const active = pending || isActive(work.job)
  const count = Number(questionCount)
  const validCount = Number.isInteger(count) && count >= 1 && count <= 50
  const result = work.job?.result
  const displayedRuns = result && !runs.some(run => run.id === result.id) ? [result, ...runs] : runs
  useEffect(() => { if (work.job && !isActive(work.job)) void Promise.allSettled([refreshRuns(), refreshDatasets()]) }, [work.job?.id, work.job?.status, refreshRuns, refreshDatasets])
  async function importDataset(file?: File) {
    setError(null); setPending(true)
    try { const imported = file ? await uploadFile<Dataset>(`/api/documents/${doc.id}/datasets`, file) : await post<Dataset>(`/api/documents/${doc.id}/datasets/sample`); setDatasetId(imported.id); await refreshDatasets() } catch (e) { setError((e as Error).message) } finally { setPending(false); if (input.current) input.current.value = '' }
  }
  async function evaluate() {
    setPending(true); setError(null)
    try {
      const path = `/api/documents/${doc.id}/evaluations${mode === 'automatic' ? '/automatic' : ''}`
      const body = mode === 'automatic' ? { question_count: count, profile, force } : { dataset_id: dataset?.id, profile, force }
      work.start(await post<JobInfo>(path, body))
      if (mode === 'automatic') setDatasetId('')
    } catch (e) { setError((e as Error).message) } finally { setPending(false) }
  }
  return <section><div className="section-intro"><h2>Automatically evaluate your RAG system</h2><p>Generate questions from this PDF, verify reference answers against source passages, then compare retrieval and answer quality.</p></div>
    <div className="panel"><h3>1. Test questions</h3><div className="evaluation-modes"><label className={`evaluation-mode ${mode === 'automatic' ? 'selected' : ''}`}><input type="radio" name="question-mode" checked={mode === 'automatic'} onChange={() => setMode('automatic')} disabled={active}/><span><strong>Generate automatically</strong><small>The system creates and source-checks the test set.</small></span></label><label className={`evaluation-mode ${mode === 'saved' ? 'selected' : ''}`}><input type="radio" name="question-mode" checked={mode === 'saved'} onChange={() => setMode('saved')} disabled={active}/><span><strong>Use a saved dataset</strong><small>Reuse generated cases or import reviewed references.</small></span></label></div>
      {mode === 'automatic' ? <><label className="field question-count">Number of test questions<input type="number" min="1" max="50" step="1" value={questionCount} onChange={e => setQuestionCount(e.target.value)} disabled={active}/></label>{!validCount ? <ErrorMessage error="Choose a whole number from 1 to 50."/> : null}<p className="muted">References are synthetic and automatically verified, with source quotes and page links. Generation uses API credits; existing verified questions can be reused.</p></> : <><p className="muted">Optional JSON or CSV import with questions, reference answers and source evidence for this PDF.</p><div className="row wrap"><button className="secondary" disabled={active} onClick={() => input.current?.click()}>Import JSON / CSV</button><button className="text-button" disabled={active} onClick={() => void importDataset()}>Use reviewed handbook sample</button><a href={`/api/documents/${doc.id}/datasets/template`}>Download template ↧</a></div><input ref={input} aria-label="Import evaluation dataset" className="visually-hidden" type="file" accept=".json,.csv" onChange={e => { if (e.target.files?.[0]) void importDataset(e.target.files[0]) }}/>{dataset ? <label className="field">Selected dataset<select value={dataset.id} disabled={active} onChange={e => setDatasetId(e.target.value)}>{datasets.map(d => <option value={d.id} key={d.id}>{d.name} · {d.count} questions</option>)}</select></label> : <p className="muted">No saved datasets yet. Generate automatically or import a dataset.</p>}</>}
    </div>
    <ErrorMessage error={error || datasetError || runsError}/>
    <div className="panel"><h3>2. Evaluation scope</h3><div className="profile-grid">{Object.entries(profiles).map(([key,[name,description]]) => <label className={`profile ${profile === key ? 'selected' : ''}`} key={key}><input type="radio" name="profile" checked={profile === key} disabled={active} onChange={() => setProfile(key as keyof typeof profiles)}/><strong>{name}</strong><span>{description}</span></label>)}</div>
      <div className="model-line">Question & answer model <strong>{config.generation_model}</strong>Reference verifier & judge <strong>{config.evaluation_model}</strong></div>
      {profile === 'full' && !config.reranker_configured ? <p className="muted">External reranking is unavailable. The report will identify the fallback to hybrid retrieval.</p> : null}
      <p className="muted">{profile === 'retrieval' ? mode === 'saved' ? 'This evaluation uses local retrieval with no model API calls.' : 'Generating and verifying questions uses API credits. Retrieval scoring runs locally.' : 'RAG answer generation and quality judging use API credits.'} Missing, failed and inapplicable scores are reported separately.</p>
      <div className="row spread evaluation-action"><label className="checkbox"><input type="checkbox" checked={force} disabled={active} onChange={e => setForce(e.target.checked)}/>Recompute cached results</label><button disabled={active || (mode === 'automatic' ? !validCount : !dataset)} onClick={() => void evaluate()}>{active ? 'Working…' : mode === 'automatic' ? 'Generate questions & evaluate →' : 'Evaluate saved questions →'}</button></div>
    </div>
    <JobProgress job={work.job} cancel={work.cancel} connectionError={work.connectionError} clear={work.clear}/>
    {dataset ? <div className="panel"><div className="row spread"><h3>Test set evidence</h3><span className="badge">{dataset.count} questions</span></div>{datasets.length > 1 && mode === 'automatic' ? <label className="field">Inspect dataset<select value={dataset.id} onChange={e => setDatasetId(e.target.value)}>{datasets.map(d => <option value={d.id} key={d.id}>{d.name}</option>)}</select></label> : null}<QuestionCases dataset={dataset} documentId={doc.id}/></div> : null}
    <div className="tip">Faithfulness measures grounding, answer relevancy measures response relevance, and context precision and recall measure useful evidence. Synthetic tests support comparison; a small generated sample does not establish quality on every question.</div>
    <div className="row spread history-title"><h2>Evaluation results</h2><span className="muted">{displayedRuns.length} runs</span></div>{displayedRuns.length ? displayedRuns.map(run => <RunCard run={run} key={run.id}/>) : <div className="panel"><p className="muted">Run an evaluation to see the strategy comparison, extra metrics and downloadable per-question results.</p></div>}
  </section>
}
