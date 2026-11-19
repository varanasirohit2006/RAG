export type IconName = 'documents' | 'evaluation' | 'results' | 'chat' | 'sources' | 'retrieval' | 'upload' | 'arrow' | 'external' | 'menu' | 'close' | 'check' | 'file'

const paths: Record<IconName, import('react').ReactNode> = {
  documents: <><path d="M3 7V5a2 2 0 0 1 2-2h5l2 3h7a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/><path d="M3 9h18"/></>,
  evaluation: <><path d="m8 3 4 7 4-7M6 3h12M9 10l-5 9a1.3 1.3 0 0 0 1 2h14a1.3 1.3 0 0 0 1-2l-5-9"/><path d="M7 16h10"/></>,
  results: <><path d="M4 3v18h17M8 16v-5m5 5V6m5 10v-8"/></>,
  chat: <><path d="M21 11a8 8 0 0 1-8 8H7l-5 3 2-6a8 8 0 0 1-1-5 8 8 0 0 1 8-8h2a8 8 0 0 1 8 8Z"/><path d="M8 9h8M8 13h5"/></>,
  sources: <><path d="M12 5a10 10 0 0 0-9-2v16a10 10 0 0 1 9 2 10 10 0 0 1 9-2V3a10 10 0 0 0-9 2Z"/><path d="M12 5v16"/></>,
  retrieval: <><circle cx="10" cy="10" r="6"/><path d="m15 15 6 6M7 10h6m-3-3v6"/></>,
  upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5"/></>,
  arrow: <path d="M4 12h16m-6-6 6 6-6 6"/>,
  external: <><path d="M14 3h7v7m0-7L10 14M10 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-5"/></>,
  menu: <path d="M4 6h16M4 12h16M4 18h16"/>,
  close: <path d="m6 6 12 12M6 18 18 6"/>,
  check: <path d="m5 12 4 4L19 6"/>,
  file: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8m-8 4h5"/></>,
}

export default function Icon({ name, className = '' }: { name: IconName; className?: string }) {
  return <svg className={`icon ${className}`} width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}
