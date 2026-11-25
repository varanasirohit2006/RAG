import { useDeferredValue, useState } from 'react'
import type { DocumentInfo } from '../types'
import Icon from '../Icons'
import { pageHref } from '../navigation'

export default function Documents({ documents, selected, select, upload, prepare, busy, maxFileMb, uploadMode, setUploadMode }: {
  documents: DocumentInfo[]; selected?: string; select: (id: string) => void;
  upload: (files?: File[]) => void; prepare: (doc: DocumentInfo) => void; busy: boolean; maxFileMb: number;
  uploadMode: 'separate' | 'combined'; setUploadMode: (mode: 'separate' | 'combined') => void;
}) {
  const [query, setQuery] = useState('')
  const [dragging, setDragging] = useState(false)
  const search = useDeferredValue(query.trim().toLowerCase())
  const filtered = documents.filter(doc => doc.name.toLowerCase().includes(search))
  return <section aria-label="Document library">
    <div className={`upload-zone ${dragging ? 'dragging' : ''}`} onDragOver={event => { event.preventDefault(); setDragging(true) }} onDragLeave={event => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false) }} onDrop={event => { event.preventDefault(); setDragging(false); if (!busy && event.dataTransfer.files.length) upload(Array.from(event.dataTransfer.files)) }}>
      <div className="upload-symbol"><Icon name="upload"/></div><div><h2>Add PDFs</h2><p>Drop or choose up to three PDFs together.</p><span className="muted">Up to {maxFileMb} MB per PDF · local OCR</span><fieldset className="row wrap"><legend className="visually-hidden">Upload mode</legend><label className="checkbox"><input type="radio" name="upload-mode" checked={uploadMode === 'separate'} disabled={busy} onChange={() => setUploadMode('separate')}/>Keep PDFs separate</label><label className="checkbox"><input type="radio" name="upload-mode" checked={uploadMode === 'combined'} disabled={busy} onChange={() => setUploadMode('combined')}/>Combine into one PDF workspace</label></fieldset></div>
      <button onClick={() => upload()} disabled={busy}>Choose PDFs <Icon name="arrow"/></button>
    </div>
    <div className="section-toolbar"><h2>Your library <span className="count-label">{documents.length}</span></h2><label className="search-field"><Icon name="retrieval"/><input aria-label="Search documents" placeholder="Search documents…" value={query} onChange={event => setQuery(event.target.value)}/></label></div>
    {filtered.length ? <div className="library-list">{filtered.map(doc => <article className={`library-row ${selected === doc.id ? 'selected' : ''}`} key={doc.id}>
      <span className="document-symbol"><Icon name="file"/></span><div className="library-document"><button className="document-title" onClick={() => select(doc.id)}>{doc.name}</button><p>{(doc.size_bytes / 1024 / 1024).toFixed(2)} MB{doc.chunk_count ? ` · ${doc.chunk_count} indexed passages` : ''}</p>{doc.preparation_error ? <p className="muted" style={{ color: 'var(--danger)', margin: '4px 0 0' }}>{doc.preparation_error}</p> : null}</div>
      <span className={`badge ${doc.status === 'ready' ? 'ready' : ''}`}>{doc.status === 'ready' ? 'Ready' : doc.status === 'preparing' ? 'Preparing' : 'Needs preparation'}{doc.source_documents ? ` · ${doc.source_documents.length} PDFs` : ''}</span>
      <div className="library-actions"><a className="icon-button" href={`/api/documents/${doc.id}/pdf`} target="_blank" rel="noreferrer" aria-label={`Open ${doc.name}`}><Icon name="external"/></a>{doc.status === 'ready' ? <a className="secondary" href={pageHref('evaluation')} onClick={() => select(doc.id)}>Evaluate <Icon name="arrow"/></a> : <button className="secondary" disabled={busy || doc.status === 'preparing'} onClick={() => { select(doc.id); prepare(doc) }}>{doc.status === 'preparing' ? 'Preparing…' : 'Prepare'}</button>}</div>
    </article>)}</div> : <div className="empty-state compact"><Icon name="documents"/><h3>{documents.length ? 'No matching documents' : 'Your library is ready for its first PDF'}</h3><p>{documents.length ? 'Try a different file name.' : 'Upload a document to generate questions, evaluate answers and inspect source evidence.'}</p></div>}
    <div className="workflow-guide"><h3>From document to evaluation</h3><ol><li><span>1</span><div><strong>Upload & prepare</strong><p>Extract the evidence and build a reusable search index.</p></div></li><li><span>2</span><div><strong>Generate & evaluate</strong><p>Create source-verified questions and test retrieval strategies.</p></div></li><li><span>3</span><div><strong>Review results</strong><p>Compare metrics, check failed cases and export your report.</p></div></li></ol></div>
  </section>
}
