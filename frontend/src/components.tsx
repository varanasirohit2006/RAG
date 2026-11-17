import ReactMarkdown from 'react-markdown'
import { Component } from 'react'
import type { ReactNode } from 'react'
import DOMPurify from 'dompurify'
import type { Answer, Job, Source } from './types'
import { isActive } from './types'

export class RuntimeBoundary extends Component<{ children: ReactNode }, { error: boolean }> {
  state = { error: false }
  static getDerivedStateFromError() { return { error: true } }
  render() {
    return this.state.error ? <main className="content"><h1>This view could not load</h1><p>Your completed documents and results are saved.</p><button onClick={() => window.location.reload()}>Reload workspace</button></main> : this.props.children
  }
}

export function ErrorMessage({ error }: { error?: Error | string | null }) {
  return error ? <div className="notice danger" role="alert">{typeof error === 'string' ? error : error.message}</div> : null
}
export function JobProgress({ job, cancel, connectionError, clear }: { job?: Job | null; cancel: () => Promise<unknown>; connectionError?: string | null; clear?: () => void }) {
  if (!job) return null
  return <section className="job-panel" aria-label={`${job.kind} progress`}>
    <div className="row spread"><strong>{job.message}</strong><span className="muted">{job.elapsed_seconds.toFixed(1)}s</span></div>
    {isActive(job) ? <><progress max={job.total} value={job.done}/><div className="row spread"><span className="muted">{job.done} / {job.total} steps</span><button className="text-button" onClick={() => { void cancel().catch(() => {}) }}>Cancel work</button></div></> : <span className={`badge ${job.status === 'succeeded' ? 'ready' : ''}`}>{job.status}</span>}
    <ErrorMessage error={job.error || connectionError}/>
    {connectionError && clear ? <button className="secondary" onClick={clear}>Dismiss unavailable job</button> : null}
    <details><summary>Stage details</summary><ol className="events">{job.events.map(([time, message], i) => <li key={i}><span>{time}s</span> {message}</li>)}</ol></details>
  </section>
}
export function Sources({ sources, documentId }: { sources: Source[]; documentId: string }) {
  return <div className="sources">{sources.map((source, index) => <details className="source" key={`${source.id}-${index}`}>
    <summary><span>Source {index+1} · {source.pages.length ? `page ${source.pages.join(', ')}` : 'document passage'}</span>{typeof source.relevance_score === 'number' ? <small>relevance {source.relevance_score.toFixed(3)}</small> : null}</summary>
    <div className="source-body"><div className="row"><code title={source.id}>{source.id.slice(0, 12)}</code>{source.pages.map(page => <a key={page} href={`/api/documents/${documentId}/pdf#page=${page}`} target="_blank" rel="noreferrer">Open page {page} ↗</a>)}</div><p className="prewrap">{source.text}</p>
      {source.tables.map((html, i) => <div key={i} className="table-scroll" dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(html, { ALLOWED_TAGS: ['table','thead','tbody','tr','th','td','caption','br'], ALLOWED_ATTR: ['colspan','rowspan'] }) }}/>) }
      {source.images.map((url, i) => <img key={url} loading="lazy" src={url} alt={`Document evidence ${i+1}`}/>)}</div>
  </details>)}</div>
}
export function AnswerCard({ answer, documentId }: { answer: Answer; documentId: string }) {
  return <article className="answer-card"><h3>{answer.query}</h3><div className="markdown"><ReactMarkdown>{answer.answer}</ReactMarkdown></div>
    <div className="timings">{Object.entries(answer.timings).map(([key, value]) => <span key={key}>{key.replaceAll('_', ' ')} <strong>{typeof value === 'number' ? value.toFixed(2) : value}{key.endsWith('seconds') ? 's' : ''}</strong></span>)}</div>
    <Sources sources={answer.sources} documentId={documentId}/></article>
}
