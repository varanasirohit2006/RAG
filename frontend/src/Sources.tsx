import DOMPurify from 'dompurify'
import type { Source } from './types'

export default function Sources({ sources, documentId }: { sources: Source[]; documentId: string }) {
  return <div className="sources">{sources.map((source, index) => <details className="source" key={`${source.id}-${index}`}>
    <summary><span className="source-heading"><strong>Passage {String(index + 1).padStart(2, '0')}</strong><span>{source.pages.length ? `Page ${source.pages.join(', ')}` : 'Document passage'}</span>{typeof source.relevance_score === 'number' ? <small>Relevance {source.relevance_score.toFixed(3)}</small> : null}</span><span className="source-excerpt">{source.text || (source.tables.length ? 'Table evidence' : 'Visual evidence')}</span></summary>
    <div className="source-body"><div className="source-links">{source.pages.map(page => <a key={page} href={`/api/documents/${documentId}/pdf#page=${page}`} target="_blank" rel="noreferrer">Open page {page} ↗</a>)}<code title={source.id}>ID {source.id.slice(0, 12)}</code></div><p className="prewrap">{source.text}</p>
      {source.original_sources?.length ? <div className="source-links">{source.original_sources.map(original => <span key={original.document_id}>{original.name}: {original.pages.map(page => <a key={page} href={`/api/documents/${original.document_id}/pdf#page=${page}`} target="_blank" rel="noreferrer">original page {page} ↗</a>)}</span>)}</div> : null}
      {source.tables.map((html, index) => <div key={index} className="table-scroll" dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(html, { ALLOWED_TAGS: ['table', 'thead', 'tbody', 'tr', 'th', 'td', 'caption', 'br'], ALLOWED_ATTR: ['colspan', 'rowspan'] }) }}/>)}
      {source.images.map((url, index) => <img key={`${url}-${index}`} loading="lazy" src={url} alt={`Document evidence ${index + 1}`}/>)}</div>
  </details>)}</div>
}
