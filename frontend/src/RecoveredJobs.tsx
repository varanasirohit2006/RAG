import { useState } from 'react'
import useSWR from 'swr'
import type { Job, JobInfo } from './types'
import { isActive } from './types'
import { post, rememberJob } from './api'
import { ErrorMessage } from './components'

const labels: Record<string, string> = { prepare: 'Document preparation', answer: 'Document answer', evaluation: 'Evaluation', evaluation_batch: 'PDF batch evaluation', retrieval: 'Retrieval experiment' }

export default function RecoveredJobs({ open, documentNames }: { open: (info: JobInfo) => void; documentNames: Record<string, string> }) {
  const { data: jobs = [], error: loadError, mutate } = useSWR<Job[]>('/api/jobs', { refreshInterval: values => values?.some(isActive) ? 1500 : 5000 })
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  function show(info: JobInfo) {
    rememberJob(info.kind === 'prepare' ? 'prepare' : `${info.kind.startsWith('evaluation') ? 'evaluation' : info.kind}:${info.document_id}`, info)
    open(info)
  }
  async function resume(job: Job) {
    setPending(job.id); setError(null)
    try { show(await post<JobInfo>(`/api/jobs/${job.id}/retry`)); await mutate() }
    catch (failure) { setError((failure as Error).message) }
    finally { setPending(null) }
  }
  if (loadError) return <ErrorMessage error={loadError}/>
  if (!jobs.length) return null
  return <section className="job-panel" aria-label="Saved work">
    <h2>Saved work</h2>
    {jobs.some(job => job.recovery_required) ? <p className="muted">The server stopped while work was in progress. Retry to reuse valid completed stages. Unfinished model requests may use API credits.</p> : null}
    <ErrorMessage error={error}/>
    {jobs.map(job => <div className="row spread wrap job-status" key={job.id}>
      <div><strong>{labels[job.kind] || job.kind}</strong><p className="muted">{documentNames[job.document_id] || 'Document'} · {job.message}</p></div>
      {job.retryable ? <button className="secondary" disabled={pending !== null} onClick={() => void resume(job)}>{pending === job.id ? 'Resuming…' : 'Resume work'}</button>
        : <button className="text-button" onClick={() => show(job)}>Open progress</button>}
    </div>)}
  </section>
}
