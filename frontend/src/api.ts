import { useEffect, useState } from 'react'
import type { Job, JobInfo } from './types'

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { 'X-RAG-Client': '1', ...init?.headers } })
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(typeof body.detail === 'string' ? body.detail : Array.isArray(body.detail) ? body.detail.map((e: { msg: string }) => e.msg).join('; ') : `Request failed (${response.status})`)
  }
  return response.json()
}
export const post = <T>(path: string, body?: unknown) => api<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
export const uploadFile = <T>(path: string, file: File) => {
  const form = new FormData(); form.append('file', file)
  return api<T>(path, { method: 'POST', body: form })
}

// Job IDs survive navigation and refresh; results and history also persist server-side.
export function useJob<T>(slot: string) {
  const storageKey = `rag-job-v1:${slot}`
  const [info, setInfo] = useState<JobInfo | null>(() => {
    try { return JSON.parse(sessionStorage.getItem(storageKey) || 'null') } catch { return null }
  })
  const [job, setJob] = useState<Job<T> | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  useEffect(() => {
    if (!info) { setJob(null); return }
    setJob({ ...info, status: 'queued', done: 0, total: 1, message: 'Connecting to job…', elapsed_seconds: 0, partial_text: '', events: [] })
    let disposed = false
    let fallback: ReturnType<typeof setInterval> | undefined
    const events = new EventSource(`/api/jobs/${info.id}/events`)
    const receive = (value: Job<T>) => {
      if (disposed) return
      setJob(value); setConnectionError(null)
      if (['succeeded', 'failed', 'cancelled'].includes(value.status)) { events.close(); if (fallback) clearInterval(fallback) }
    }
    events.onmessage = event => receive(JSON.parse(event.data))
    events.onerror = () => {
      events.close()
      const poll = () => api<Job<T>>(`/api/jobs/${info.id}`).then(receive).catch(error => {
        if (!disposed) { setConnectionError(error.message); if (fallback) clearInterval(fallback) }
      })
      if (!fallback) { fallback = setInterval(poll, 1500); void poll() }
    }
    return () => { disposed = true; events.close(); if (fallback) clearInterval(fallback) }
  }, [info])
  const start = (value: JobInfo) => { sessionStorage.setItem(storageKey, JSON.stringify(value)); setConnectionError(null); setInfo(value) }
  const clear = () => { sessionStorage.removeItem(storageKey); setInfo(null); setConnectionError(null) }
  return { job, start, clear, connectionError, cancel: () => info ? post(`/api/jobs/${info.id}/cancel`) : Promise.resolve() }
}
