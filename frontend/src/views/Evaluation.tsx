import { useEffect, useRef, useState } from 'react'
import useSWR from 'swr'
import type { BatchEvaluationResult, Config, Dataset, DocumentInfo, JobInfo, Run } from '../types'
import { isActive } from '../types'
import { post, rememberJob, uploadFile, useJob } from '../api'
import { ErrorMessage, JobProgress } from '../components'
import Icon from '../Icons'
import EvaluationQuestions from '../EvaluationQuestions'
import { profiles, qualityMetrics, metricDescriptions, metricLabel } from '../evaluationData'
import type { Profile } from '../evaluationData'
import { pageHref } from '../navigation'
import PdfScope from '../PdfScope'
import type { PdfScopeMode } from '../PdfScope'

export default function EvaluationView({ doc, config, documents, selectDocument }: { doc: DocumentInfo; config: Config; documents: DocumentInfo[]; selectDocument: (id: string) => Promise<void> }) {
  const { data: datasets = [], error: datasetError, mutate: refreshDatasets } = useSWR<Dataset[]>(`/api/documents/${doc.id}/datasets`)
  const { mutate: refreshRuns } = useSWR<Run[]>(`/api/documents/${doc.id}/runs`)
  const work = useJob<Run | BatchEvaluationResult>(`evaluation:${doc.id}`)
  const [pdfScope, setPdfScope] = useState<PdfScopeMode>('current')
  const [selectedPdfs, setSelectedPdfs] = useState<string[]>(() => doc.source_documents?.map(source => source.document_id) || [doc.id])
  const [mode, setMode] = useState<'automatic' | 'saved'>('automatic')
  const [datasetId, setDatasetId] = useState('')
  const [questionCount, setQuestionCount] = useState('10')
  const [profile, setProfile] = useState<Profile>('full')
  const [force, setForce] = useState(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)
  const dataset = datasets.find(d => d.id === datasetId) || datasets[0]
  const active = pending || isActive(work.job)
  const count = Number(questionCount)
  const validCount = Number.isInteger(count) && count >= 1 && count <= 50
  const validSelection = pdfScope === 'current' || selectedPdfs.length >= 2 && selectedPdfs.length <= 3
  const cases = mode === 'automatic' ? validCount && validSelection ? count * (pdfScope === 'separate' ? selectedPdfs.length : 1) : null : dataset?.count
  const selectedProfile = profiles[profile]
  useEffect(() => { if (work.job && !isActive(work.job)) void Promise.allSettled([refreshRuns(), refreshDatasets()]) }, [work.job?.id, work.job?.status, refreshRuns, refreshDatasets])
  useEffect(() => {
    const setup = work.job?.evaluation_setup
    if (!setup) return
    setQuestionCount(String(setup.question_count)); setProfile(setup.profile); setSelectedPdfs(setup.document_ids)
    setPdfScope(doc.source_documents?.length ? 'current' : setup.mode); setMode('automatic')
  }, [work.job?.id, work.job?.evaluation_setup?.question_count])
  const batchResult = work.job?.status === 'succeeded' && work.job.result && 'batch_results' in work.job.result ? work.job.result : null

  async function importDataset(file: File) {
    setError(null); setPending(true)
    try { const imported = await uploadFile<Dataset>(`/api/documents/${doc.id}/datasets`, file); setDatasetId(imported.id); await refreshDatasets() } catch (e) { setError((e as Error).message) } finally { setPending(false); if (input.current) input.current.value = '' }
  }
  async function evaluate() {
    setPending(true); setError(null)
    try {
      if (pdfScope !== 'current') {
        const info = await post<JobInfo>('/api/evaluations/batch', { document_ids: selectedPdfs, mode: pdfScope, question_count: count, profile, force })
        rememberJob(`evaluation:${info.document_id}`, info)
        if (info.document_id === doc.id) work.start(info)
        else await selectDocument(info.document_id)
        return
      }
      const path = `/api/documents/${doc.id}/evaluations${mode === 'automatic' ? '/automatic' : ''}`
      const body = mode === 'automatic' ? { question_count: count, profile, force } : { dataset_id: dataset?.id, profile, force }
      work.start(await post<JobInfo>(path, body))
      if (mode === 'automatic') setDatasetId('')
    } catch (e) { setError((e as Error).message) } finally { setPending(false) }
  }

  return <section aria-label="Evaluation setup">
    <PdfScope mode={pdfScope} setMode={value => { setPdfScope(value); if (value !== 'current') setMode('automatic') }} documents={documents} selected={selectedPdfs} setSelected={setSelectedPdfs} disabled={active}/>
    {!validSelection ? <ErrorMessage error="Choose two or three original PDFs for this evaluation."/> : null}
    <ol className="steps"><li className="done"><span><Icon name="check"/></span><div><strong>Document ready</strong><small>Evidence is indexed</small></div></li><li className="current"><span>2</span><div><strong>Configure test</strong><small>Questions and scope</small></div></li><li><span>3</span><div><strong>Review results</strong><small>Metrics and evidence</small></div></li></ol>
    <ErrorMessage error={error || datasetError}/>
    <div className="evaluation-layout">
      <div className="setup-panel">
        <div className="setup-section"><div className="section-title"><span className="step-number">1</span><div><h2>Test questions</h2><p>Choose how to build the test set for this document.</p></div></div>
          <fieldset className="segmented-options"><legend className="visually-hidden">Question source</legend><label className={mode === 'automatic' ? 'selected' : ''}><input type="radio" name="question-mode" checked={mode === 'automatic'} onChange={() => setMode('automatic')} disabled={active}/><strong>Generate automatically</strong></label><label className={mode === 'saved' ? 'selected' : ''}><input type="radio" name="question-mode" checked={mode === 'saved'} onChange={() => setMode('saved')} disabled={active || pdfScope !== 'current'}/><strong>Use saved questions</strong></label></fieldset>
          {mode === 'automatic' ? <><div className="question-count-row"><label className="field" htmlFor="question-count">Number of questions<input id="question-count" type="number" min="1" max="50" step="1" value={questionCount} onChange={e => setQuestionCount(e.target.value)} disabled={active} aria-describedby="question-count-help"/></label><p id="question-count-help">Questions are drafted from source passages. Each reference answer is checked against supporting quotes{config.independent_reference_verifier === false ? ' using the same configured model; this is not independent verification.' : ' by a separately configured judge.'}</p></div>{!validCount ? <ErrorMessage error="Choose a whole number from 1 to 50."/> : null}<p className="inline-note"><Icon name="sources"/>You can inspect every generated question, reference and source after the run.</p></> : <><p className="muted">Reuse a generated test set, or import source-labelled questions as JSON or CSV.</p>{dataset ? <label className="field">Dataset<select value={dataset.id} disabled={active} onChange={e => setDatasetId(e.target.value)}>{datasets.map(d => <option value={d.id} key={d.id}>{d.name} · {d.count} questions</option>)}</select></label> : <div className="notice">No saved questions yet. Import a dataset or generate questions automatically.</div>}<div className="row wrap"><button className="secondary" disabled={active} onClick={() => input.current?.click()}><Icon name="upload"/>Import JSON / CSV</button><a className="text-link" href={`/api/documents/${doc.id}/datasets/template`}>Download template ↗</a></div><input ref={input} aria-label="Import evaluation dataset" className="visually-hidden" type="file" accept=".json,.csv" onChange={e => { if (e.target.files?.[0]) void importDataset(e.target.files[0]) }}/></>}
        </div>
        <div className="setup-section"><div className="section-title"><span className="step-number">2</span><div><h2>Evaluation scope</h2><p>Choose the comparison that answers your next question.</p></div></div>
          <fieldset className="profile-grid"><legend className="visually-hidden">Evaluation profile</legend>{Object.entries(profiles).map(([key, value]) => <label className={`profile ${profile === key ? 'selected' : ''}`} key={key}><input type="radio" name="profile" checked={profile === key} disabled={active} onChange={() => setProfile(key as Profile)}/><strong>{value.label}</strong><small>{key === 'full' ? '4 strategies · quality + retrieval' : key === 'preview' ? 'Production pipeline · selected subset' : '3 strategies · evidence metrics'}</small></label>)}</fieldset>
          <p className="profile-description">{selectedProfile.description}</p>
          {profile === 'full' && !config.reranker_configured ? <div className="notice">Reranking is not configured. Its results will explicitly show the hybrid-retrieval fallback.</div> : null}
          <details className="technical-details"><summary>Models & cache settings</summary><dl className="model-settings"><dt>Questions & answers</dt><dd>{config.generation_model}</dd><dt>Reference verifier & judge</dt><dd>{config.evaluation_model}</dd></dl><label className="checkbox"><input type="checkbox" checked={force} disabled={active} onChange={e => setForce(e.target.checked)}/>Recompute instead of reusing cached results</label></details>
        </div>
      </div>
      <aside className="run-summary" aria-label="Run summary"><h2>This evaluation</h2><dl><div><dt>Test questions</dt><dd>{cases ?? '—'}</dd></div><div><dt>Retrieval strategies</dt><dd>{selectedProfile.strategies}</dd></div><div><dt>Answer-quality metrics</dt><dd>{profile === 'retrieval' ? 'Not included' : '5'}</dd></div></dl>{profile === 'preview' ? <p className="muted">Preview scores a limited subset of the test set. The report records the actual question count.</p> : null}
        <div className="run-summary-note"><strong>Source evidence comes first</strong><p>Unsupported references are rejected before evaluation. Failed and unavailable scores stay visible in the report.</p></div>
        <p className="api-note">{profile === 'retrieval' ? mode === 'saved' ? 'Scoring runs locally with no model API calls.' : 'Drafting and verifying questions use model APIs. Retrieval scoring is local.' : 'Question generation, answers and quality judging use model APIs. Each question can require multiple judge requests.'}</p>
        {pdfScope === 'separate' ? <p className="muted">{selectedPdfs.length} PDFs × {validCount ? count : '—'} questions per PDF. Model API work applies to every PDF.</p> : null}
        <button className="primary-action" disabled={active || !validSelection || (mode === 'automatic' ? !validCount : !dataset)} onClick={() => void evaluate()}>{active ? 'Evaluation in progress…' : mode === 'automatic' ? 'Generate & evaluate' : 'Evaluate saved questions'}<Icon name="arrow"/></button>
        <span className="summary-footnote">Completed results appear on the Results page.</span>
      </aside>
    </div>
    <JobProgress job={work.job} cancel={work.cancel} connectionError={work.connectionError} clear={work.clear} retry={work.retry}/>
    {batchResult ? <section className="job-panel" aria-label="PDF batch reports"><h2>PDF batch finished · {batchResult.status}</h2>{batchResult.batch_results.map(result => <div className="row spread wrap" key={result.document_id}><strong>{result.name}</strong>{result.run ? <button className="secondary" onClick={() => { void selectDocument(result.document_id).then(() => { window.location.hash = pageHref('results') }) }}>Open report · {String(result.run.manifest.status)}</button> : <ErrorMessage error={result.error || 'No report was produced.'}/>}</div>)}</section> : work.job?.status === 'succeeded' ? <div className="completion-banner"><div><Icon name="check"/><span><strong>Your report is ready</strong><small>Review strategy scores, test questions and any failed metrics.</small></span></div><a className="secondary" href={pageHref('results')}>Open results <Icon name="arrow"/></a></div> : null}
    {dataset ? <details className="evidence-panel"><summary><Icon name="sources"/><span>Saved test questions</span><span className="count-label">{dataset.count}</span></summary><div className="evidence-panel-body">{datasets.length > 1 ? <label className="field">Inspect dataset<select value={dataset.id} onChange={e => setDatasetId(e.target.value)}>{datasets.map(d => <option value={d.id} key={d.id}>{d.name}</option>)}</select></label> : null}<EvaluationQuestions key={dataset.id} dataset={dataset} documentId={doc.id}/></div></details> : null}
    <details className="metric-guide"><summary>What do the five quality metrics measure?</summary><div className="metric-definitions">{qualityMetrics.map(name => <div key={name}><strong>{metricLabel(name)}</strong><p>{metricDescriptions[name]}</p></div>)}</div><p className="muted">Synthetic tests support comparison. A small generated sample does not establish quality on every question.</p></details>
  </section>
}
