import { useEffect, useState } from 'react'

export const pages = {
  documents: { title: 'Documents', description: 'Upload PDFs and prepare their text, tables and visual evidence for evaluation.' },
  evaluation: { title: 'Evaluation', description: 'Generate source-verified questions and measure your RAG system automatically.' },
  results: { title: 'Results', description: 'Compare retrieval strategies, inspect failures and review the evidence behind every run.' },
  chat: { title: 'Ask document', description: 'Ask a question and read an answer grounded in your document, with linked sources.' },
  sources: { title: 'Source explorer', description: 'Browse the text, tables and images available to your RAG system.' },
  retrieval: { title: 'Retrieval lab', description: 'Explore how keyword, vector and hybrid search select evidence for a query.' },
} as const
export type Page = keyof typeof pages
export const pageHref = (page: Page) => `#/${page}`

function currentPage(): Page {
  const name = window.location.hash.slice(2).split('?')[0].split('/')[0]
  return name in pages ? name as Page : 'evaluation'
}

export function usePage() {
  const [page, setPage] = useState(currentPage)
  useEffect(() => {
    const update = () => setPage(currentPage())
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])
  useEffect(() => { document.title = `${pages[page].title} · RAG Workspace` }, [page])
  return page
}

export function readStored(key: string, fallback = '') {
  try { return localStorage.getItem(key) || fallback } catch { return fallback }
}
export function saveStored(key: string, value: string) {
  try { localStorage.setItem(key, value) } catch { /* A restricted browser can still use the workspace. */ }
}

export function formatDate(value?: string) {
  if (!value) return 'Date unavailable'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? 'Date unavailable' : date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
