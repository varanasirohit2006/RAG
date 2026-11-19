import { useState } from 'react'
import useSWR from 'swr'
import type { DocumentInfo, Source } from '../types'
import { ErrorMessage } from '../components'
import Sources from '../Sources'

export default function Chunks({ doc }: { doc: DocumentInfo }) {
  const [offset, setOffset] = useState(0)
  const { data, error, isLoading } = useSWR<{total: number; items: Source[]}>(`/api/documents/${doc.id}/chunks?offset=${offset}&limit=12`)
  return <section aria-label="Source passages"><ErrorMessage error={error}/>
    <div className="row spread pagination"><span>{data ? data.total ? `${offset+1}–${Math.min(offset+12, data.total)} of ${data.total} passages` : 'No indexed passages' : 'Loading passages…'}</span><div className="row"><button className="secondary" disabled={offset === 0 || isLoading} onClick={() => setOffset(n => Math.max(0,n-12))}>Previous</button><button className="secondary" disabled={!data || offset+12 >= data.total || isLoading} onClick={() => setOffset(n => n+12)}>Next</button></div></div>
    {isLoading ? <p className="loading" role="status">Loading document passages…</p> : data?.items.length ? <Sources sources={data.items} documentId={doc.id}/> : data ? <div className="empty-state compact"><h2>No passages available</h2><p>Prepare a document with readable text or scanned evidence to browse its sources.</p></div> : null}
  </section>
}
