import { Component, useState } from 'react'
import type { ReactNode } from 'react'
import type { Job } from './types'
import { isActive } from './types'
import Icon from './Icons'

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

export function JobProgress({ job, cancel, connectionError, clear, retry }: { job?: Job | null; cancel: () => Promise<unknown>; connectionError?: string | null; clear?: () => void; retry?: () => Promise<unknown> }) {
  const [cancelError, setCancelError] = useState<string | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const [retrying, setRetrying] = useState(false)
  if (!job) return null
  async function cancelJob() {
    setCancelling(true); setCancelError(null)
    try { await cancel() } catch (error) { setCancelError((error as Error).message) } finally { setCancelling(false) }
  }
  async function retryJob() {
    setRetrying(true); setCancelError(null)
    try { await retry?.() } catch (error) { setCancelError((error as Error).message) } finally { setRetrying(false) }
  }
  return <section className="job-panel" aria-label={`${job.kind} progress`}>
    <div className="row spread"><div className="job-message" role="status" aria-live="polite"><span className={`job-indicator ${isActive(job) ? 'working' : ''}`}/><strong>{job.message}</strong></div><span className="muted">{job.elapsed_seconds.toFixed(1)}s</span></div>
    {isActive(job) ? <><progress aria-label="Job progress" max={job.total} value={job.done}/><div className="row spread"><span className="muted">{job.done} / {job.total} steps</span><button className="text-button" disabled={cancelling} onClick={() => void cancelJob()}>{cancelling ? 'Requesting cancellation…' : 'Cancel work'}</button></div></> : <div className="row spread job-status"><span className={`badge ${job.status === 'succeeded' ? 'ready' : 'warning'}`}>{job.status}</span>{clear ? <button className="text-button" onClick={clear}>Dismiss</button> : null}</div>}
    <ErrorMessage error={job.error || connectionError || cancelError}/>
    {job.retryable && retry ? <button className="secondary" disabled={retrying} onClick={() => void retryJob()}>{retrying ? 'Resuming…' : 'Retry interrupted work'}</button> : null}
    {connectionError && clear ? <button className="secondary" onClick={clear}>Dismiss unavailable job</button> : null}
    <details className="technical-details"><summary>Stage details</summary><ol className="events">{job.events.map(([time, message], index) => <li key={index}><span>{time}s</span> {message}</li>)}</ol></details>
  </section>
}

export function SectionHeading({ title, description }: { title: string; description: string }) {
  return <div className="section-title"><Icon name="sources"/><div><h2>{title}</h2><p>{description}</p></div></div>
}
