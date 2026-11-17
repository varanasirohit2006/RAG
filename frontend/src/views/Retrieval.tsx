import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import type { DocumentInfo, JobInfo, RetrievalResult } from '../types'
import { isActive } from '../types'
import { post, useJob } from '../api'
import { ErrorMessage, JobProgress, Sources } from '../components'

export default function Retrieval({ doc }: { doc: DocumentInfo }) {
  const work = useJob<RetrievalResult>(`retrieval:${doc.id}`)
  const [question, setQuestion] = useState('')
  const [variations, setVariations] = useState('')
  const [strategy, setStrategy] = useState('hybrid')
  const [weight, setWeight] = useState(0.5)
  const [k, setK] = useState(10)
  const [topN, setTopN] = useState(3)
  const [expand, setExpand] = useState(false)
  const [rerank, setRerank] = useState(false)
  const [generate, setGenerate] = useState(false)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  async function search() {
    setError(null); setPending(true)
    try { work.start(await post<JobInfo>(`/api/documents/${doc.id}/retrieval`, { question, variations: variations.split('\n').filter(q => q.trim()), strategy, vector_weight: weight, k, top_n: topN, expand, rerank, generate })) } catch (e) { setError((e as Error).message) } finally { setPending(false) }
  }
  const result = work.job?.status === 'succeeded' ? work.job.result : null
  return <section><div className="section-intro"><h2>Retrieval lab</h2><p>Compare vector, keyword, and weighted hybrid retrieval. Combine query variations with reciprocal rank fusion.</p></div>
    <form className="panel" onSubmit={e => { e.preventDefault(); void search() }}><label className="field">Search query<input required maxLength={4000} value={question} onChange={e => setQuestion(e.target.value)} placeholder="Find evidence in this document"/></label>
      <div className="settings-grid"><label className="field">Strategy<select value={strategy} onChange={e => setStrategy(e.target.value)}><option value="hybrid">Weighted hybrid</option><option value="vector">Vector search</option><option value="keyword">Keyword search</option></select></label><label className="field">Retrieved passages (K)<input type="number" min={1} max={30} value={k} onChange={e => setK(Number(e.target.value))}/></label><label className="field">Final passages<input type="number" min={1} max={Math.min(k,10)} value={topN} onChange={e => setTopN(Number(e.target.value))}/></label></div>
      {strategy === 'hybrid' ? <label className="field">Vector {Math.round(weight*100)}% · keyword {Math.round((1-weight)*100)}%<input type="range" min={0} max={1} step={0.1} value={weight} onChange={e => setWeight(Number(e.target.value))}/></label> : null}
      <details><summary>Query variations & optional model calls</summary><label className="field">Manual variations (up to 3, one per line)<textarea rows={3} value={variations} onChange={e => setVariations(e.target.value)}/></label><div className="row wrap"><label className="checkbox"><input type="checkbox" checked={expand} onChange={e => setExpand(e.target.checked)}/>Generate variations</label><label className="checkbox"><input type="checkbox" checked={rerank} onChange={e => setRerank(e.target.checked)}/>Rerank</label><label className="checkbox"><input type="checkbox" checked={generate} onChange={e => setGenerate(e.target.checked)}/>Generate answer</label></div><p className="muted">These options use configured model APIs. Default retrieval runs locally.</p></details>
      <div className="row end"><button disabled={!question.trim() || pending || isActive(work.job)}>{pending || isActive(work.job) ? 'Searching…' : 'Search document →'}</button></div></form>
    <ErrorMessage error={error}/><JobProgress job={work.job} cancel={work.cancel} connectionError={work.connectionError} clear={work.clear}/>
    {result ? <div className="panel"><h3>Fused evidence · {result.timings.total_seconds.toFixed(2)}s</h3>{result.warning ? <div className="notice">{result.warning}</div> : null}<Sources sources={result.sources} documentId={doc.id}/>{result.answer ? <div className="markdown"><ReactMarkdown>{result.answer}</ReactMarkdown></div> : null}<details><summary>Results by query ({result.queries.length})</summary>{result.per_query.map((q,i) => <div key={i}><h3>{q.query}</h3><Sources sources={q.sources} documentId={doc.id}/></div>)}</details></div> : null}
  </section>
}
