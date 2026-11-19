import { useEffect, useState } from 'react'
import useSWR from 'swr'
import type { BatchEvaluationResult, Dataset, DocumentInfo, Run } from '../types'
import { isActive } from '../types'
import { useJob } from '../api'
import { ErrorMessage, JobProgress } from '../components'
import Icon from '../Icons'
import EvaluationQuestions from '../EvaluationQuestions'
import { display, metricDescriptions, metricLabel, profiles, qualityMetrics, retrievalMetrics } from '../evaluationData'
import type { MetricDefinition, Profile } from '../evaluationData'
import { formatDate, pageHref } from '../navigation'

type SummaryRow = Run['summary'][number]
type ResultTab = 'quality' | 'retrieval' | 'diagnostics' | 'latency' | 'questions' | 'details'
const tabLabels: Record<ResultTab, string> = { quality: 'Answer quality', retrieval: 'Retrieval', diagnostics: 'Citations & diagnostics', latency: 'Latency & reliability', questions: 'Test questions', details: 'Run details' }

function MetricTable({ rows, metrics, catalog, detailed }: { rows: SummaryRow[]; metrics: string[]; catalog: Record<string, MetricDefinition>; detailed: boolean }) {
  return <div className="table-scroll" tabIndex={0} role="region" aria-label="Strategy scores"><table className="metric-table"><caption>Mean valid scores by strategy. Each metric uses its own applicable samples.</caption><thead><tr><th scope="col">Strategy</th>{metrics.map(name => <th scope="col" key={name} title={catalog[name]?.description || metricDescriptions[name]}>{metricLabel(name, catalog)}</th>)}</tr></thead>
    <tbody>{rows.map((row, index) => <tr key={`${row.strategy}-${index}`}><th scope="row"><span className="strategy-dot"/>{display(row.strategy)}</th>{metrics.map(name => {
      const count = row[`${name}_count`]
      const failed = Number(row[`${name}_failed`] || 0), skipped = Number(row[`${name}_skipped`] || 0)
      const blocked = Number(row[`${name}_blocked`] || 0)
      const lower = row[`${name}_lower95`], upper = row[`${name}_upper95`]
      const interval = typeof lower === 'number' && typeof upper === 'number' ? `${display(lower)}–${display(upper)}` : null
      return <td key={name}><strong className={row[name] == null ? 'missing-score' : ''}>{display(row[name])}</strong><small>{typeof count === 'number' ? `${count} / ${display(row.samples)} scored` : row[name] == null ? 'Unavailable' : 'Count unavailable'}</small>{failed ? <small className="metric-failure">{failed} failed</small> : null}{blocked ? <small className="metric-failure">{blocked} blocked by provider</small> : null}{skipped > blocked ? <small>{skipped - blocked} skipped / N/A</small> : null}{detailed && interval ? <small className="score-interval">95% CI {interval}</small> : null}</td>
    })}</tr>)}</tbody></table></div>
}

function RunReport({ run, dataset, documentId }: { run: Run; dataset?: Dataset; documentId: string }) {
  const [selectedTab, setSelectedTab] = useState<ResultTab>('quality')
  const [detailed, setDetailed] = useState(false)
  const m = run.manifest
  const catalog = (m.metric_catalog && typeof m.metric_catalog === 'object' && !Array.isArray(m.metric_catalog) ? m.metric_catalog : {}) as Record<string, MetricDefinition>
  const columns = Array.from(new Set(run.summary.flatMap(row => Object.keys(row))))
  const hasQuality = qualityMetrics.some(name => columns.includes(name))
  const retrieval = retrievalMetrics.filter(name => columns.includes(name))
  const diagnostics = columns.filter(name => !qualityMetrics.includes(name) && !retrievalMetrics.includes(name) && columns.includes(`${name}_count`) && !name.endsWith('_seconds'))
  const operational = ['samples', 'completed', 'failed', 'skipped', 'failure_rate', 'partial_rate', 'error_rate', 'sample_seconds_p50', 'sample_seconds_p95', 'retrieval_seconds_p50', 'retrieval_seconds_p95', 'generation_seconds_p50', 'generation_seconds_p95', 'judge_seconds_p50', 'judge_seconds_p95'].filter(name => columns.includes(name))
  const available: ResultTab[] = [...(hasQuality ? ['quality' as const] : []), ...(retrieval.length ? ['retrieval' as const] : []), ...(diagnostics.length ? ['diagnostics' as const] : []), ...(operational.length ? ['latency' as const] : []), 'questions', 'details']
  const tab = available.includes(selectedTab) ? selectedTab : available[0]
  const metrics = tab === 'quality' ? qualityMetrics : tab === 'retrieval' ? retrieval : diagnostics
  const warnings = [...new Set([...(Array.isArray(m.warnings) ? m.warnings : []), ...(Array.isArray(m.dataset_warnings) ? m.dataset_warnings : [])].map(String))]
  const status = String(m.status || 'unknown')
  const profile = profiles[m.profile as Profile]?.label || display(m.profile)
  const partialScores = columns.filter(name => name.endsWith('_failed')).some(name => run.summary.some(row => Number(row[name] || 0) > 0))
  const providerLimit = m.provider_limit && typeof m.provider_limit === 'object' ? m.provider_limit as { reason?: string; reset_at?: string } : null
  return <article className="report-panel">
    <header className="report-header"><div><h2>{profile}</h2><p>{formatDate(run.created_at)} · {display(m.sampled_questions)} of {display(m.total_questions)} questions</p></div><span className={`badge ${status === 'completed' && !partialScores ? 'ready' : 'warning'}`}>{status.replaceAll('_', ' ')}</span></header>
    <dl className="report-stats"><div><dt>Completed samples</dt><dd>{display(m.completed_samples)}</dd></div><div><dt>Failed / partial</dt><dd className={Number(m.failed_samples) > 0 ? 'metric-failure' : ''}>{display(m.failed_samples)}</dd></div><div><dt>Skipped samples</dt><dd>{display(m.skipped_samples)}</dd></div><div><dt>Run time</dt><dd>{typeof m.seconds === 'number' ? `${m.seconds.toFixed(1)}s` : '—'}</dd></div></dl>
    {run.cache_reused ? <div className="report-notice">This report reuses completed cached evaluation work.</div> : null}
    {partialScores ? <div className="notice">Some metrics failed. Compare valid sample counts before interpreting scores; unavailable scores do not mean zero.</div> : null}
    {run.automatic && run.automatic.accepted_questions !== run.automatic.requested_questions ? <div className="notice">{run.automatic.accepted_questions} of {run.automatic.requested_questions} requested questions passed source verification.</div> : null}
    <ErrorMessage error={m.error ? String(m.error) : null}/>
    {providerLimit ? <div className="notice danger" role="alert"><strong>Provider blocked this run</strong><p>{providerLimit.reason || 'The provider rejected further requests.'}</p>{providerLimit.reset_at ? <p>Reported reset: {new Date(providerLimit.reset_at).toLocaleString()}</p> : null}</div> : null}
    {run.metric_failures?.length ? <details className="report-warnings"><summary>Why {run.metric_failures.length} metric attempts failed</summary><div className="table-scroll" tabIndex={0} role="region" aria-label="Metric failure details"><table><thead><tr><th>Strategy / question</th><th>Metric</th><th>Exact cause</th></tr></thead><tbody>{run.metric_failures.map((failure, index) => <tr key={index}><td>{failure.strategy}<br/>{failure.question_id}</td><td>{metricLabel(failure.metric)}</td><td>{failure.reason}</td></tr>)}</tbody></table></div></details> : null}
    {warnings.length ? <details className="report-warnings"><summary>{warnings.length} run warning{warnings.length === 1 ? '' : 's'}</summary>{warnings.map((warning, index) => <p key={index}>{warning}</p>)}</details> : null}
    <div className="result-tabs" aria-label="Report sections">{available.map(name => <button key={name} className={tab === name ? 'active' : ''} aria-pressed={tab === name} onClick={() => setSelectedTab(name)}>{tabLabels[name]}</button>)}</div>
    <section className="report-section" aria-label={tabLabels[tab]}>
      {tab === 'quality' || tab === 'retrieval' || tab === 'diagnostics' ? <><div className="section-toolbar"><div><h3>{tab === 'quality' ? 'Answer quality comparison' : tab === 'retrieval' ? 'Evidence retrieval' : 'Citations & additional diagnostics'}</h3><p className="muted">{tab === 'quality' ? 'Read grounding, relevance and correctness separately.' : tab === 'retrieval' ? 'Scores use the test set’s labelled source evidence.' : 'Structural citation and wording checks complement the quality metrics.'}</p></div><label className="checkbox"><input type="checkbox" checked={detailed} onChange={event => setDetailed(event.target.checked)}/>Show confidence intervals</label></div><MetricTable rows={run.summary} metrics={metrics} catalog={catalog} detailed={detailed}/>
        <details className="metric-guide"><summary>Metric definitions & interpretation</summary><div className="metric-definitions">{metrics.map(name => { const definition = catalog[name]; return <div key={name}><strong>{metricLabel(name, catalog)}</strong><p>{definition?.description || metricDescriptions[name] || 'See the run manifest for this metric’s method and applicability.'}</p>{definition ? <small>{definition.method ? `Method: ${definition.method}. ` : ''}{definition.range ? `Range: ${JSON.stringify(definition.range)}. ` : ''}{definition.direction ? `Direction: ${definition.direction}. ` : ''}{definition.applicability ? `Applies to: ${typeof definition.applicability === 'string' ? definition.applicability : JSON.stringify(definition.applicability)}` : ''}</small> : null}</div> })}</div><p className="muted">Citation structure and lexical overlap do not prove semantic support or factual correctness. Generated references may be incomplete; small samples are exploratory.</p></details>
      </> : tab === 'latency' ? <><h3>Latency & reliability by strategy</h3><p className="muted">Times are in seconds; rates are fractions of samples. Percentiles describe this run, and cached timings differ from fresh execution.</p><div className="table-scroll" tabIndex={0} role="region" aria-label="Latency and reliability"><table><caption>Measured stage latency and sample outcomes.</caption><thead><tr><th scope="col">Strategy</th>{operational.map(name => <th scope="col" key={name}>{metricLabel(name)}</th>)}</tr></thead><tbody>{run.summary.map((row, index) => <tr key={index}><th scope="row">{display(row.strategy)}</th>{operational.map(name => <td key={name}>{display(row[name])}</td>)}</tr>)}</tbody></table></div></> : tab === 'questions' ? dataset ? <EvaluationQuestions key={dataset.id} dataset={dataset} documentId={documentId}/> : <div className="empty-state compact"><h3>Test set unavailable</h3><p>This report’s dataset is not available. Export the per-sample results to inspect its saved question records.</p></div> : <><h3>Configuration & methodology</h3><p className="muted">Run identifiers, models, rubric versions, warnings and recorded execution settings.</p><pre className="manifest">{JSON.stringify({ ...m, automatic: run.automatic, dataset_id: run.dataset_id }, null, 2)}</pre></>}
    </section>
    <footer className="report-exports"><span>Export this run</span><div>{(['summary', 'samples', 'manifest'] as const).map(kind => <a className="secondary" key={kind} href={`/api/runs/${run.id}/export/${kind}`}>{kind === 'summary' ? 'Summary CSV' : kind === 'samples' ? 'Sample results' : 'Run manifest'} <Icon name="arrow"/></a>)}</div></footer>
  </article>
}

export default function Results({ doc }: { doc: DocumentInfo }) {
  const { data: runs = [], error, isLoading, mutate } = useSWR<Run[]>(`/api/documents/${doc.id}/runs`)
  const { data: datasets = [], error: datasetError, mutate: refreshDatasets } = useSWR<Dataset[]>(`/api/documents/${doc.id}/datasets`)
  const work = useJob<Run | BatchEvaluationResult>(`evaluation:${doc.id}`)
  const [selected, setSelected] = useState('')
  const latest = work.job?.status === 'succeeded' && work.job.result && 'summary' in work.job.result ? work.job.result : undefined
  const all = (latest && !runs.some(run => run.id === latest.id) ? [latest, ...runs] : runs).slice().sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at))
  const run = all.find(item => item.id === selected) || all[0]
  useEffect(() => { if (work.job && !isActive(work.job)) void Promise.allSettled([mutate(), refreshDatasets()]) }, [work.job?.id, work.job?.status, mutate, refreshDatasets])
  return <section aria-label="Evaluation results"><ErrorMessage error={error || datasetError}/><JobProgress job={work.job?.status === 'succeeded' ? null : work.job} cancel={work.cancel} connectionError={work.connectionError} clear={work.clear}/>
    {isLoading && !all.length ? <p className="loading" role="status">Loading evaluation reports…</p> : !run ? <div className="empty-state"><div className="empty-symbol"><Icon name="results"/></div><h2>No evaluation reports yet</h2><p>Run an automatic evaluation to compare strategies,<br className="desktop-break"/> inspect source-verified questions and download the results.</p><a className="primary-link" href={pageHref('evaluation')}>Set up an evaluation <Icon name="arrow"/></a></div> : <div className="results-layout"><aside className="run-history" aria-label="Run history"><div className="section-toolbar"><h2>Run history</h2><span className="count-label">{all.length}</span></div>{all.map(item => <button className={`run-history-item ${run.id === item.id ? 'selected' : ''}`} key={item.id} onClick={() => setSelected(item.id)} aria-pressed={run.id === item.id}><div><strong>{profiles[item.manifest.profile as Profile]?.label || display(item.manifest.profile)}</strong><span className={`run-status-dot ${item.manifest.status === 'completed' ? 'ready' : ''}`}/></div><time>{formatDate(item.created_at)}</time><span>{display(item.manifest.sampled_questions)} questions · {display(item.manifest.status).replaceAll('_', ' ')}</span></button>)}<a className="secondary" href={pageHref('evaluation')}>New evaluation <Icon name="arrow"/></a></aside><RunReport key={run.id} run={run} dataset={datasets.find(dataset => dataset.id === run.dataset_id)} documentId={doc.id}/></div>}
  </section>
}
