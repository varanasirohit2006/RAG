import { useState } from 'react'
import useSWR from 'swr'
import type { DocumentInfo, Source } from '../types'
import { ErrorMessage, Sources } from '../components'

export default function Chunks({ doc }: { doc: DocumentInfo }) {
  const [offset, setOffset] = useState(0)
  const { data, error, isLoading } = useSWR<{total: number; items: Source[]}>(`/api/documents/${doc.id}/chunks?offset=${offset}&limit=12`)
  return <section><div className="section-intro"><h2>Source explorer</h2><p>Inspect the original text, tables, and images used for retrieval. Chunk IDs can label your evaluation dataset.</p></div><ErrorMessage error={error}/>
    <div className="row spread pagination"><span>{data ? `${offset+1}–${Math.min(offset+12, data.total)} of ${data.total} passages` : 'Loading passages…'}</span><div className="row"><button className="secondary" disabled={offset === 0 || isLoading} onClick={() => setOffset(n => Math.max(0,n-12))}>Previous</button><button className="secondary" disabled={!data || offset+12 >= data.total || isLoading} onClick={() => setOffset(n => n+12)}>Next</button></div></div>
    {isLoading ? <p className="loading">Loading document passages…</p> : data ? <Sources sources={data.items} documentId={doc.id}/> : null}
  </section>
}
