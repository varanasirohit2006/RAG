import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import useSWR from 'swr'
import type { Config, DocumentInfo, JobInfo } from './types'
import { isActive } from './types'
import { post, uploadFile, useJob } from './api'
import { ErrorMessage, JobProgress } from './components'
import Chat from './views/Chat'

const Chunks = lazy(() => import('./views/Chunks'))
const Evaluation = lazy(() => import('./views/Evaluation'))
const Retrieval = lazy(() => import('./views/Retrieval'))
const tabs = ['Evaluation', 'Ask document', 'Source explorer', 'Retrieval lab'] as const

export default function App() {
  const { data: config, error: configError } = useSWR<Config>('/api/bootstrap')
  const { data: documents = [], error: documentsError, mutate } = useSWR<DocumentInfo[]>(config ? '/api/documents' : null)
  const [selected, setSelected] = useState(() => localStorage.getItem('rag-document-v1') || '')
  const [tab, setTab] = useState<(typeof tabs)[number]>('Evaluation')
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const input = useRef<HTMLInputElement>(null)
  const prep = useJob<DocumentInfo>('prepare')
  const doc = documents.find(d => d.id === selected) || documents.find(d => d.status === 'ready') || documents[0]
  useEffect(() => { if (doc) localStorage.setItem('rag-document-v1', doc.id) }, [doc?.id])
  useEffect(() => { if (prep.job?.status === 'succeeded' || prep.job?.status === 'failed' || prep.job?.status === 'cancelled') void mutate() }, [prep.job?.id, prep.job?.status, mutate])

  async function upload(file: File) {
    setError(null); setUploading(true)
    try {
      if (file.size > (config?.max_file_mb || 50)*1024*1024) throw new Error(`PDF must be under ${config?.max_file_mb || 50} MB.`)
      const result = await uploadFile<{document: DocumentInfo; job: JobInfo | null}>('/api/documents', file)
      setSelected(result.document.id); setTab('Evaluation')
      if (result.job) prep.start(result.job)
      await mutate()
    } catch (e) { setError((e as Error).message) } finally { setUploading(false); if (input.current) input.current.value = '' }
  }
  async function prepare() {
    if (!doc) return
    setError(null)
    try { prep.start(await post<JobInfo>(`/api/documents/${doc.id}/prepare`)) } catch (e) { setError((e as Error).message) }
  }
  return <div className="app-shell">
    <aside className="sidebar"><a className="brand" href="/" aria-label="RAG Workspace home"><span className="brand-icon">R</span><span>RAG<span className="brand-light"> Workspace</span></span></a>
      <div className="workspace-label">DOCUMENT LIBRARY</div>
      <button className="upload-button" disabled={!config || uploading || isActive(prep.job)} onClick={() => input.current?.click()}>{uploading ? 'Uploading…' : '+ Upload PDF'}</button>
      <input ref={input} type="file" accept=".pdf,application/pdf" aria-label="Upload PDF" className="visually-hidden" onChange={e => { if (e.target.files?.[0]) void upload(e.target.files[0]) }}/>
      <p className="sidebar-caption">Text, scans, tables & charts<br/>Up to {config?.max_file_mb || 50} MB · local OCR</p>
      <div className="document-list">{documents.map(d => <button key={d.id} className={`document-button ${doc?.id === d.id ? 'selected' : ''}`} onClick={() => setSelected(d.id)}><span className="file-icon">PDF</span><span className="document-name">{d.name}<small>{(d.size_bytes/1024/1024).toFixed(2)} MB · {d.status === 'ready' ? 'Ready' : 'Needs preparation'}</small></span></button>)}</div>
      <div className="sidebar-footer"><span className="status-dot"/> Local workspace<p>Documents stay on your computer.<br/>Model requests use your API credits.</p></div>
    </aside>
    <main className="main"><header className="topbar"><span>DOCUMENT INTELLIGENCE</span><span className="badge">Python API · React</span></header>
      <div className="content"><ErrorMessage error={configError || documentsError || error}/>
      <div className="page-heading"><div><div className="eyebrow">GENERATE. TEST. MEASURE.</div><h1>{doc ? doc.name : 'Evaluate your RAG system.'}</h1><p className="muted">Automatic questions, source-backed references and retrieval comparisons.</p></div>{doc ? <a className="secondary" href={`/api/documents/${doc.id}/pdf`} target="_blank" rel="noreferrer">Open PDF ↗</a> : null}</div>
      <JobProgress job={prep.job} cancel={prep.cancel} connectionError={prep.connectionError} clear={prep.clear}/>
      {!config && !configError ? <div className="empty-state">Connecting to the document service…</div> : !doc ? <section className="empty-state"><span className="empty-icon">↥</span><h2>Start with a PDF</h2><p>Upload a document. The system generates test questions,<br/>checks their references and measures RAG quality.</p><button onClick={() => input.current?.click()} disabled={uploading}>{uploading ? 'Uploading…' : 'Choose a PDF'}</button></section> : <>
        <div className="document-meta"><span className={`badge ${doc.status === 'ready' ? 'ready' : ''}`}>{doc.status === 'ready' ? 'Ready to explore' : 'Preparation required'}</span>{Object.entries(doc.stats).map(([category,count]) => <span key={category}>{count} {category}</span>)}{doc.chunk_count ? <span>{doc.chunk_count} indexed passages</span> : null}<code title={doc.sha256}>SHA-256 {doc.sha256.slice(0, 12)}…</code></div>
        <nav className="tabs" aria-label="Workspace views">{tabs.map(name => <button key={name} aria-current={tab === name ? 'page' : undefined} className={tab === name ? 'active' : ''} onClick={() => setTab(name)}>{name}</button>)}</nav>
        {doc.status !== 'ready' ? <div className="empty-state"><h2>Prepare this document</h2><p>Extract readable text and visual evidence, then create its local search index.</p><button disabled={isActive(prep.job)} onClick={() => void prepare()}>Prepare document</button></div> : <Suspense fallback={<p className="loading">Loading workspace view…</p>}>
          {tab === 'Ask document' ? <Chat key={doc.id} doc={doc} config={config!}/> : tab === 'Source explorer' ? <Chunks key={doc.id} doc={doc}/> : tab === 'Evaluation' ? <Evaluation key={doc.id} doc={doc} config={config!}/> : <Retrieval key={doc.id} doc={doc}/>}
        </Suspense>}
      </>}
      </div>
    </main>
  </div>
}
