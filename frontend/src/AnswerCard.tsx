import ReactMarkdown from 'react-markdown'
import type { Answer } from './types'
import Sources from './Sources'
import { formatDate } from './navigation'

export default function AnswerCard({ answer, documentId }: { answer: Answer; documentId: string }) {
  const sources = answer.sources || []
  return <article className="answer-card"><div className="answer-heading"><h3>{answer.query}</h3>{answer.created_at ? <time>{formatDate(answer.created_at)}</time> : null}</div><div className="markdown"><ReactMarkdown>{answer.answer}</ReactMarkdown></div>
    <details className="technical-details answer-timings"><summary>Response timing</summary><div className="timings">{Object.entries(answer.timings || {}).map(([key, value]) => <span key={key}>{key.replaceAll('_', ' ')} <strong>{typeof value === 'number' ? value.toFixed(2) : value}{key.endsWith('seconds') ? 's' : ''}</strong></span>)}</div></details>
    <h4>Supporting evidence <span className="count-label">{sources.length}</span></h4><Sources sources={sources} documentId={documentId}/></article>
}
