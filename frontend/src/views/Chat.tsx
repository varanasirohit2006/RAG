import { useDeferredValue, useEffect, useState } from 'react'
import useSWR from 'swr'
import ReactMarkdown from 'react-markdown'
import type { Answer, Config, DocumentInfo, JobInfo } from '../types'
import { isActive } from '../types'
import { post, useJob } from '../api'
import { ErrorMessage, JobProgress } from '../components'
import AnswerCard from '../AnswerCard'
import Icon from '../Icons'

export default function Chat({ doc, config }: { doc: DocumentInfo; config: Config }) {
  const { data: answers = [], error: historyError, mutate } = useSWR<Answer[]>(`/api/documents/${doc.id}/answers`)
  const work = useJob<Answer>(`answer:${doc.id}`)
  const [question, setQuestion] = useState('')
  const [rerank, setRerank] = useState(() => {
    try {
      const stored = localStorage.getItem('rag-rerank')
      return stored !== null ? stored === 'true' && config.reranker_configured : config.reranker_configured
    } catch { return config.reranker_configured }
  })
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const partial = useDeferredValue(work.job?.partial_text || '')
  const completedResult = work.job?.status === 'succeeded' && work.job.result ? work.job.result : null
  const allAnswers = completedResult && !answers.some(a => a.id === completedResult.id) ? [completedResult, ...answers] : answers
  useEffect(() => { if (work.job?.status === 'succeeded') void mutate() }, [work.job?.id, work.job?.status, mutate])
  async function ask() {
    setError(null); setSending(true)
    try { work.start(await post<JobInfo>(`/api/documents/${doc.id}/answers`, { question, rerank })); setQuestion('') } catch (e) { setError((e as Error).message) } finally { setSending(false) }
  }
  const busy = sending || isActive(work.job)
  return <section aria-label="Document questions">
    <form className="question-box" onSubmit={e => { e.preventDefault(); void ask() }}><label htmlFor="question">Your question</label><textarea id="question" placeholder="What would you like to know about this document?" value={question} maxLength={4000} onChange={e => setQuestion(e.target.value)} rows={3}/><div className="row spread"><label className="checkbox"><input type="checkbox" checked={rerank} disabled={!config.reranker_configured || busy} onChange={e => { setRerank(e.target.checked); try { localStorage.setItem('rag-rerank', String(e.target.checked)) } catch { /* ignore */ } }}/>Rerank evidence{!config.reranker_configured ? ' (unavailable)' : ''}</label><button disabled={busy || !question.trim()} type="submit">{busy ? 'Working…' : 'Ask document'}<Icon name="arrow"/></button></div></form>
    <div className="model-line">Answer model <strong>{config.generation_model}</strong><span>Local hybrid search · cited sources</span></div>
    <ErrorMessage error={error || historyError}/><JobProgress job={work.job} cancel={work.cancel} connectionError={work.connectionError} clear={work.clear} retry={work.retry}/>
    {partial && work.job?.status !== 'succeeded' ? <article className="answer-card"><p className="small-label">{isActive(work.job) ? 'Answer streaming' : 'Incomplete answer'}</p><div className="markdown"><ReactMarkdown>{partial}</ReactMarkdown></div></article> : null}
    {allAnswers.length ? <div className="row spread history-title"><h2>Conversation history</h2><span className="muted">{allAnswers.length} answers</span></div> : <div className="tip">Try asking for a definition, a comparison, or an explanation of a table. You can open the original evidence with every answer.</div>}
    {allAnswers.map(answer => <AnswerCard key={answer.id} answer={answer} documentId={doc.id}/>)}
  </section>
}
