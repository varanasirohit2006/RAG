import { useDeferredValue, useEffect, useState } from 'react'
import useSWR from 'swr'
import ReactMarkdown from 'react-markdown'
import type { Answer, Config, DocumentInfo, JobInfo } from '../types'
import { isActive } from '../types'
import { post, useJob } from '../api'
import { AnswerCard, ErrorMessage, JobProgress } from '../components'

export default function Chat({ doc, config }: { doc: DocumentInfo; config: Config }) {
  const { data: answers = [], error: historyError, mutate } = useSWR<Answer[]>(`/api/documents/${doc.id}/answers`)
  const work = useJob<Answer>(`answer:${doc.id}`)
  const [question, setQuestion] = useState('')
  const [rerank, setRerank] = useState(config.reranker_configured)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const partial = useDeferredValue(work.job?.partial_text || '')
  useEffect(() => { if (work.job?.status === 'succeeded') void mutate() }, [work.job?.id, work.job?.status, mutate])
  async function ask() {
    setError(null); setSending(true)
    try { work.start(await post<JobInfo>(`/api/documents/${doc.id}/answers`, { question, rerank })); setQuestion('') } catch (e) { setError((e as Error).message) } finally { setSending(false) }
  }
  const busy = sending || isActive(work.job)
  return <section><div className="section-intro"><h2>Ask your document</h2><p>Answers use retrieved passages, tables, and available visual evidence.</p></div>
    <form className="question-box" onSubmit={e => { e.preventDefault(); void ask() }}><label htmlFor="question">Your question</label><textarea id="question" placeholder="What would you like to know about this document?" value={question} maxLength={4000} onChange={e => setQuestion(e.target.value)} rows={3}/><div className="row spread"><label className="checkbox"><input type="checkbox" checked={rerank} onChange={e => setRerank(e.target.checked)}/>Rerank evidence</label><button disabled={busy || !question.trim()} type="submit">{busy ? 'Working…' : 'Ask document →'}</button></div></form>
    <div className="model-line">Answer model <strong>{config.generation_model}</strong><span>Local hybrid search · cited sources</span></div>
    <ErrorMessage error={error || historyError}/><JobProgress job={work.job} cancel={work.cancel} connectionError={work.connectionError} clear={work.clear}/>
    {partial && work.job?.status !== 'succeeded' ? <article className="answer-card"><span className="eyebrow">{isActive(work.job) ? 'ANSWER STREAMING' : 'INCOMPLETE ANSWER'}</span><div className="markdown"><ReactMarkdown>{partial}</ReactMarkdown></div></article> : null}
    {answers.length ? <div className="row spread history-title"><h2>Conversation history</h2><span className="muted">{answers.length} answers</span></div> : <div className="tip">Try asking for a definition, a comparison, or an explanation of a table. You can open the original evidence with every answer.</div>}
    {answers.map(answer => <AnswerCard key={answer.id} answer={answer} documentId={doc.id}/>)}
  </section>
}
