import { lazy, Suspense, useEffect, useRef, useState } from 'react'
import useSWR from 'swr'
import type { AuthSession, BatchUpload, Config, DocumentInfo, JobInfo } from './types'
import { isActive } from './types'
import { post, uploadFiles, useJob } from './api'
import { ErrorMessage, JobProgress } from './components'
import Icon from './Icons'
import { pageHref, pages, readStored, saveStored, usePage } from './navigation'
import type { Page } from './navigation'
import Documents from './views/Documents'
import RecoveredJobs from './RecoveredJobs'

const Evaluation = lazy(() => import('./views/Evaluation'))
const Results = lazy(() => import('./views/Results'))
const Chat = lazy(() => import('./views/Chat'))
const Chunks = lazy(() => import('./views/Chunks'))
const Retrieval = lazy(() => import('./views/Retrieval'))
const groups: { label: string; items: Page[] }[] = [
  { label: 'Workspace', items: ['documents', 'evaluation', 'results'] },
  { label: 'Explore', items: ['chat', 'sources', 'retrieval'] },
]

export default function App({ account }: { account: AuthSession }) {
  const { data: config, error: configError } = useSWR<Config>('/api/bootstrap')
  const { data: documents = [], error: documentsError, isLoading: documentsLoading, mutate } = useSWR<DocumentInfo[]>(config ? '/api/documents' : null, { refreshInterval: values => values?.some(value => value.status === 'preparing') ? 1000 : 0 })
  const [selected, setSelected] = useState(() => readStored('rag-document-v1'))
  const page = usePage()
  const [menuOpen, setMenuOpen] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [uploading, setUploading] = useState(false)
  const [preparing, setPreparing] = useState(false)
  const [uploadMode, setUploadMode] = useState<'separate' | 'combined'>('separate')
  const input = useRef<HTMLInputElement>(null)
  const heading = useRef<HTMLHeadingElement>(null)
  const menuButton = useRef<HTMLButtonElement>(null)
  const prep = useJob<DocumentInfo>('prepare')
  const doc = documents.find(d => d.id === selected) || documents.find(d => d.status === 'ready') || documents[0]
  const busy = uploading || preparing || isActive(prep.job) || doc?.status === 'preparing'
  const loading = !config && !configError || !!config && documentsLoading
  useEffect(() => { if (doc) saveStored('rag-document-v1', doc.id) }, [doc?.id])
  useEffect(() => { if (prep.job && !isActive(prep.job)) void mutate() }, [prep.job?.id, prep.job?.status, mutate])
  useEffect(() => {
    setMenuOpen(false)
    heading.current?.focus({ preventScroll: true })
    window.scrollTo({ top: 0 })
  }, [page])
  useEffect(() => {
    if (!menuOpen) return
    const drawer = document.getElementById('workspace-navigation')!
    const controls = Array.from(drawer.querySelectorAll<HTMLAnchorElement | HTMLButtonElement>('a[href], button:not(:disabled)'))
    controls.find(control => control.getAttribute('aria-current') === 'page')?.focus()
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setMenuOpen(false); event.preventDefault() }
      if (event.key === 'Tab') {
        const first = controls[0], last = controls[controls.length - 1]
        if (event.shiftKey && document.activeElement === first) { last.focus(); event.preventDefault() }
        else if (!event.shiftKey && document.activeElement === last) { first.focus(); event.preventDefault() }
      }
    }
    window.addEventListener('keydown', keyboard)
    return () => { window.removeEventListener('keydown', keyboard); menuButton.current?.focus() }
  }, [menuOpen])

  async function upload(files: File[]) {
    setError(null); setUploading(true)
    try {
      if (!files.length || files.length > (config?.max_batch_pdfs || 3)) throw new Error(`Choose 1–${config?.max_batch_pdfs || 3} PDFs at a time.`)
      if (uploadMode === 'combined' && files.length < 2) throw new Error('Choose at least two PDFs to combine, or select separate PDFs.')
      for (const file of files) {
        if (!file.name.toLowerCase().endsWith('.pdf')) throw new Error(`${file.name}: choose a PDF document.`)
        if (file.size > (config?.max_file_mb || 50) * 1024 * 1024) throw new Error(`${file.name}: PDF must be under ${config?.max_file_mb || 50} MB.`)
      }
      const result = await uploadFiles<BatchUpload>('/api/documents/batch', files, uploadMode)
      const selectedDocument = result.combined || result.documents[0]
      if (selectedDocument) setSelected(selectedDocument.id)
      if (result.job) prep.start(result.job)
      if (result.errors.length) setError(result.errors.map(error => `${error.name}: ${error.error}`).join('; '))
      await mutate()
      window.location.hash = pageHref(files.length > 1 && !result.combined || result.errors.length ? 'documents' : 'evaluation')
    } catch (e) { setError((e as Error).message); void mutate() } finally { setUploading(false); if (input.current) input.current.value = '' }
  }
  function chooseFile(files?: File[]) { if (files?.length) void upload(files); else input.current?.click() }
  async function selectDocument(id: string) { await mutate(); setSelected(id) }
  async function prepare(target: DocumentInfo) {
    setError(null); setPreparing(true)
    try { prep.start(await post<JobInfo>(`/api/documents/${target.id}/prepare`)) } catch (e) { setError((e as Error).message) } finally { setPreparing(false) }
  }
  async function signOut() {
    try { await post('/api/auth/logout'); window.location.assign('/') } catch (e) { setError((e as Error).message) }
  }

  return <div className="app-shell">
    <a className="skip-link" href="#main-content" onClick={event => { event.preventDefault(); document.getElementById('main-content')?.focus() }}>Skip to content</a>
    {menuOpen ? <button className="navigation-backdrop" aria-label="Close navigation" onClick={() => setMenuOpen(false)}/> : null}
    <aside className={`sidebar ${menuOpen ? 'open' : ''}`} aria-label="Workspace navigation" id="workspace-navigation" role={menuOpen ? 'dialog' : undefined} aria-modal={menuOpen || undefined}>
      <a className="brand" href={pageHref('evaluation')}><span className="brand-icon">R</span><span>RAG<span className="brand-light">Workspace</span></span></a>
      <button className="icon-button close-menu" aria-label="Close navigation" onClick={() => setMenuOpen(false)}><Icon name="close"/></button>
      <nav>{groups.map(group => <div className="nav-group" key={group.label}><p className="nav-label">{group.label}</p>{group.items.map(name => <a key={name} href={pageHref(name)} className={`nav-item ${page === name ? 'active' : ''}`} aria-current={page === name ? 'page' : undefined} onClick={() => setMenuOpen(false)}><Icon name={name}/><span>{pages[name].title}</span></a>)}</div>)}</nav>
      <div className="sidebar-bottom"><button className="upload-button" disabled={!config || busy} onClick={() => chooseFile()}><Icon name="upload"/>{uploading ? 'Uploading…' : 'Upload PDF'}</button><div className="workspace-note"><span className="status-dot"/>{account.mode === 'oidc' ? 'Private workspace' : 'Local workspace'}<p>{account.user ? <span className="account-name">{account.user.name}</span> : <>Files are stored locally.<br/>Model calls use your API credits.</>}</p></div>{account.user ? <button className="secondary sign-out" onClick={() => void signOut()}>Sign out</button> : null}</div>
    </aside>
    <div className="main" inert={menuOpen}>
      <header className="topbar"><div className="breadcrumb"><button ref={menuButton} className="icon-button menu-button" aria-label="Open navigation" aria-expanded={menuOpen} aria-controls="workspace-navigation" onClick={() => setMenuOpen(true)}><Icon name="menu"/></button><span>Workspace</span><span aria-hidden="true">/</span><strong>{pages[page].title}</strong></div><div className="document-switcher"><label htmlFor="active-document">Document</label><select id="active-document" value={doc?.id || ''} disabled={!documents.length} onChange={event => setSelected(event.target.value)}>{documents.length ? documents.map(d => <option value={d.id} key={d.id}>{d.name}</option>) : <option value="">No document selected</option>}</select>{doc ? <a className="icon-button" href={`/api/documents/${doc.id}/pdf`} target="_blank" rel="noreferrer" aria-label="Open selected PDF"><Icon name="external"/></a> : null}</div></header>
      <main id="main-content" className="content" tabIndex={-1}>
        <div className="page-heading"><div><h1 ref={heading} tabIndex={-1}>{pages[page].title}</h1><p>{pages[page].description}</p></div>{page === 'evaluation' ? <a className="secondary" href={pageHref('results')}>View results <Icon name="arrow"/></a> : null}</div>
        <ErrorMessage error={configError || documentsError || error}/>
        {config ? <RecoveredJobs documentNames={Object.fromEntries(documents.map(document => [document.id, document.name]))} open={info => { setSelected(info.document_id); window.location.hash = pageHref(info.kind === 'answer' ? 'chat' : info.kind === 'retrieval' ? 'retrieval' : 'evaluation') }}/> : null}
        <JobProgress job={prep.job} cancel={prep.cancel} connectionError={prep.connectionError} clear={prep.clear} retry={prep.retry}/>
        <input ref={input} type="file" multiple accept=".pdf,application/pdf" aria-label="Upload PDFs" className="visually-hidden" onChange={event => { if (event.target.files?.length) chooseFile(Array.from(event.target.files)) }}/>
        {loading ? <div className="loading" role="status">Connecting to your workspace…</div> : !config || documentsError ? <div className="empty-state compact"><h2>Workspace unavailable</h2><p>Check that the Python service is running, then reload the page.</p><button onClick={() => window.location.reload()}>Reload workspace</button></div> : page === 'documents' ? <Documents documents={documents} selected={doc?.id} select={setSelected} upload={chooseFile} prepare={target => void prepare(target)} busy={busy} maxFileMb={config.max_file_mb} uploadMode={uploadMode} setUploadMode={setUploadMode}/> : !doc ? <section className="empty-state"><div className="empty-symbol"><Icon name="documents"/></div><h2>Start with a document</h2><p>Upload a PDF. We’ll prepare the evidence, generate test questions<br className="desktop-break"/> and help you measure how well your RAG system answers them.</p><button disabled={busy} onClick={() => chooseFile()}>Upload your first PDF <Icon name="arrow"/></button><a className="text-link" href={pageHref('documents')}>How the document workflow works</a></section> : <>
          <div className="document-context"><div><Icon name="file"/><span className="context-name">{doc.name}</span></div><div><span className={`badge ${doc.status === 'ready' ? 'ready' : ''}`}>{doc.status === 'ready' ? 'Ready' : 'Needs preparation'}</span>{doc.chunk_count ? <span className="muted">{doc.chunk_count} passages</span> : null}</div></div>
          {doc.status !== 'ready' && page !== 'results' ? <div className="empty-state"><div className="empty-symbol"><Icon name="evaluation"/></div><h2>{isActive(prep.job) && prep.job?.document_id === doc.id ? 'Preparing your document' : 'Prepare this document'}</h2><p>Extract readable text, tables and visual evidence,<br className="desktop-break"/> then create the local index used by evaluation and search.</p><button disabled={busy} onClick={() => void prepare(doc)}>{busy ? 'Preparation in progress…' : 'Prepare document'}</button></div> : <Suspense fallback={<p className="loading" role="status">Loading {pages[page].title.toLowerCase()}…</p>}>
            {page === 'evaluation' ? <Evaluation key={doc.id} doc={doc} documents={documents} selectDocument={selectDocument} config={config}/> : page === 'results' ? <Results key={doc.id} doc={doc}/> : page === 'chat' ? <Chat key={doc.id} doc={doc} config={config}/> : page === 'sources' ? <Chunks key={doc.id} doc={doc}/> : <Retrieval key={doc.id} doc={doc}/>}
          </Suspense>}
        </>}
        <footer className="page-footer"><span>RAG Workspace</span><span>Source evidence. Transparent evaluation.</span></footer>
      </main>
    </div>
  </div>
}
