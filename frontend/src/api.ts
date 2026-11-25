import { useEffect, useState } from 'react'
import type { Job, JobInfo } from './types'

class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message) }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { 'X-RAG-Client': '1', ...init?.headers } })
  if (!response.ok) {
    if (response.status === 401) window.dispatchEvent(new Event('rag-auth-required'))
    const body = await response.json().catch(() => ({}))
    throw new ApiError(typeof body.detail === 'string' ? body.detail : Array.isArray(body.detail) ? body.detail.map((e: { msg: string }) => e.msg).join('; ') : `Request failed (${response.status})`, response.status)
  }
  return response.json()
}
export const post = <T>(path: string, body?: unknown) => api<T>(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
export const uploadFile = <T>(path: string, file: File) => {
  const form = new FormData(); form.append('file', file)
  return api<T>(path, { method: 'POST', body: form })
}
export const uploadFiles = <T>(path: string, files: File[], mode: string) => {
  const form = new FormData()
  files.forEach(file => form.append('files', file))
  form.append('mode', mode)
  return api<T>(path, { method: 'POST', body: form })
}

// Job IDs survive navigation and refresh; results and history also persist server-side.
export function rememberJob(slot: string, info: JobInfo) {
  try { sessionStorage.setItem(`rag-job-v1:${slot}`, JSON.stringify(info)) } catch { /* Progress still works without browser storage. */ }
  window.dispatchEvent(new CustomEvent('rag-job-start', { detail: { slot, info } }))
}

export function useJob<T>(slot: string) {
  const storageKey = `rag-job-v1:${slot}`
  const [info, setInfo] = useState<JobInfo | null>(() => {
    try {
      const value = JSON.parse(sessionStorage.getItem(storageKey) || 'null')
      return value && /^[a-f0-9]{32}$/.test(value.id) && /^[a-f0-9]{32}$/.test(value.document_id) && typeof value.kind === 'string' ? value : null
    } catch { return null }
  })
  const [job, setJob] = useState<Job<T> | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)
  useEffect(() => {
    try {
      const value = JSON.parse(sessionStorage.getItem(storageKey) || 'null')
      setInfo(value && /^[a-f0-9]{32}$/.test(value.id) && /^[a-f0-9]{32}$/.test(value.document_id) && typeof value.kind === 'string' ? value : null)
    } catch { setInfo(null) }
  }, [storageKey])
  useEffect(() => {
    const receive = (event: Event) => {
      const detail = (event as CustomEvent<{ slot: string; info: JobInfo }>).detail
      if (detail.slot === slot) { setInfo(detail.info); setConnectionError(null) }
    }
    window.addEventListener('rag-job-start', receive)
    return () => window.removeEventListener('rag-job-start', receive)
  }, [slot])
  useEffect(() => {
    if (!info) { setJob(null); return }
    setJob({ ...info, status: 'queued', done: 0, total: 1, message: 'Connecting to job…', elapsed_seconds: 0, partial_text: '', events: [] })
    let disposed = false
    let polling = false
    let fallback: ReturnType<typeof setInterval> | undefined
    const events = new EventSource(`/api/jobs/${info.id}/events`)
    const receive = (value: Job<T>) => {
      if (disposed) return
      setJob(value); setConnectionError(null)
      if (['succeeded', 'failed', 'cancelled'].includes(value.status)) { events.close(); if (fallback) clearInterval(fallback) }
    }
    events.onmessage = event => {
      try { receive(JSON.parse(event.data)) }
      catch { if (!disposed) setConnectionError('Malformed job event received') }
    }
    events.onerror = () => {
      events.close()
      const poll = async () => {
        if (disposed || polling) return
        polling = true
        try { receive(await api<Job<T>>(`/api/jobs/${info.id}`)) } catch (error) {
          if (!disposed) {
            setConnectionError((error as Error).message)
            if (error instanceof ApiError && [401, 403, 404].includes(error.status) && fallback) clearInterval(fallback)
          }
        } finally { polling = false }
      }
      if (!fallback) { fallback = setInterval(poll, 1500); void poll() }
    }
    return () => { disposed = true; events.close(); if (fallback) clearInterval(fallback) }
  }, [info])
  const start = (value: JobInfo) => { rememberJob(slot, value); setConnectionError(null); setInfo(value) }
  const clear = () => { try { sessionStorage.removeItem(storageKey) } catch { /* Optional browser storage. */ } setInfo(null); setConnectionError(null) }
  return { job, start, clear, connectionError, cancel: () => info ? post(`/api/jobs/${info.id}/cancel`) : Promise.resolve(),
    retry: async () => { if (info) start(await post<JobInfo>(`/api/jobs/${info.id}/retry`)) } }
}
