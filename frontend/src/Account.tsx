import { useEffect, useState } from 'react'
import useSWR, { SWRConfig } from 'swr'
import type { AuthSession } from './types'
import App from './App'
import { ErrorMessage } from './components'

export default function Account() {
  const { data: session, error, mutate } = useSWR<AuthSession>('/api/auth/session', { refreshInterval: 30000 })
  const [ready, setReady] = useState('')
  const [signInFailed] = useState(() => new URLSearchParams(window.location.search).has('auth_error'))
  const identity = session?.authenticated ? session.user?.id || 'local' : ''
  useEffect(() => {
    const query = new URLSearchParams(window.location.search)
    if (query.has('auth_error')) {
      query.delete('auth_error')
      window.history.replaceState(null, '', window.location.pathname + (query.size ? '?' + query : '') + window.location.hash)
    }
  }, [])
  useEffect(() => {
    if (identity) {
      try {
        const previous = sessionStorage.getItem('rag-account-v1')
        if (previous !== identity && (previous !== null || identity !== 'local')) {
          for (const key of Object.keys(sessionStorage)) if (key.startsWith('rag-job-v1:')) sessionStorage.removeItem(key)
        }
        sessionStorage.setItem('rag-account-v1', identity)
      } catch { /* Server-side saved work remains available without browser storage. */ }
    }
    setReady(identity)
  }, [identity])
  useEffect(() => {
    const expired = () => { void mutate(value => value?.mode === 'oidc' ? { ...value, authenticated: false, user: null } : value, { revalidate: true }) }
    window.addEventListener('rag-auth-required', expired)
    return () => window.removeEventListener('rag-auth-required', expired)
  }, [mutate])
  if (session?.authenticated && ready === identity) {
    return <SWRConfig key={identity} value={{ provider: () => new Map() }}><App account={session}/></SWRConfig>
  }
  return <main className="account-page">
    <section className="account-card" aria-label="Account access">
      <span className="brand-icon" aria-hidden="true">R</span>
      <h1>RAG Workspace</h1>
      {!session && !error || session?.authenticated ? <p role="status">Connecting to your workspace…</p> : error ? <><ErrorMessage error={error}/><button onClick={() => void mutate()}>Try again</button></> : <>
        <p>Sign in to open your private documents, evaluations and saved work.</p>
        {signInFailed ? <p role="alert" className="error">Sign-in could not be completed. Please try again.</p> : null}
        <a className="primary-link" href="/api/auth/login">Sign in</a>
      </>}
    </section>
  </main>
}
