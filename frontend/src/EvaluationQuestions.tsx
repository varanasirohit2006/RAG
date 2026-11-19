import { useDeferredValue, useState } from 'react'
import type { Dataset } from './types'
import Icon from './Icons'

export default function EvaluationQuestions({ dataset, documentId }: { dataset: Dataset; documentId: string }) {
  const [search, setSearch] = useState('')
  const query = useDeferredValue(search.toLowerCase().trim())
  const synthetic = dataset.origin === 'synthetic' || dataset.provenance?.origin === 'synthetic'
  const cases = dataset.cases.filter(item => `${item.question} ${item.reference_answer}`.toLowerCase().includes(query))
  return <div className="question-evidence">
    <div className="section-toolbar"><div><h3>{dataset.count} {synthetic ? 'generated' : 'saved'} questions</h3><p className="muted">{synthetic ? 'References are automatically source-verified; they are not human-reviewed ground truth.' : 'Imported references are validated for document identity and evidence labels.'}</p></div><a className="secondary" href={`/api/datasets/${dataset.id}/export`}>Export questions <Icon name="arrow"/></a></div>
    {dataset.warnings?.map((warning, index) => <div className="notice" key={index}>{warning}</div>)}
    <label className="search-field question-search"><Icon name="retrieval"/><input aria-label="Search test questions" placeholder="Search questions or references…" value={search} onChange={event => setSearch(event.target.value)}/></label>
    <div className="question-list">{cases.map((item, index) => <details className="question-case" key={item.id}>
      <summary><span className="question-number">{String(index + 1).padStart(2, '0')}</span><span>{item.question}</span>{item.category ? <span className="question-category">{item.category.replaceAll('_', ' ')}</span> : null}</summary>
      <div className="question-case-body"><h4>Reference answer</h4><p>{item.reference_answer}</p><div className="row wrap"><strong className="small-label">{item.answerable ? 'Supporting evidence' : 'Unanswerable case'}</strong>{item.reference_pages.map(page => <a key={page} href={`/api/documents/${documentId}/pdf#page=${page}`} target="_blank" rel="noreferrer">Page {page} ↗</a>)}{item.relevant_chunk_ids?.length ? <span className="muted">{item.relevant_chunk_ids.length} labelled passages</span> : null}</div>
        {item.supporting_quotes?.map((quote, quoteIndex) => <blockquote key={quoteIndex}>{typeof quote === 'string' ? quote : quote.quote || quote.text || ''}</blockquote>)}
        {item.verification ? <details className="technical-details"><summary>Verification details</summary><pre>{JSON.stringify(item.verification, null, 2)}</pre></details> : null}
      </div>
    </details>)}</div>{!cases.length ? <p className="muted">No questions match your search.</p> : null}
    {dataset.provenance ? <details className="technical-details"><summary>Dataset provenance</summary><pre>{JSON.stringify(dataset.provenance, null, 2)}</pre></details> : null}
  </div>
}
