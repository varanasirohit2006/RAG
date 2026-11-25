import type { DocumentInfo } from './types'

export type PdfScopeMode = 'current' | 'separate' | 'combined'

export default function PdfScope({ mode, setMode, documents, selected, setSelected, disabled }: {
  mode: PdfScopeMode; setMode: (mode: PdfScopeMode) => void; documents: DocumentInfo[];
  selected: string[]; setSelected: (ids: string[]) => void; disabled: boolean;
}) {
  const originals = documents.filter(document => !document.source_documents?.length)
  return <section className="setup-section" aria-label="PDF evaluation scope">
    <h2>PDF scope</h2><p className="muted">Evaluate the selected workspace, each chosen PDF separately, or one combined collection.</p>
    <fieldset className="profile-grid"><legend className="visually-hidden">PDF evaluation mode</legend>{([
      ['current', 'Selected PDF', 'One report for the current workspace'],
      ['separate', 'Each PDF separately', 'Independent questions and report per PDF'],
      ['combined', 'Combined PDFs', 'One set of questions across all chosen PDFs'],
    ] as const).map(([value, label, help]) => {
      const isUnavailable = value !== 'current' && originals.length < 2
      return <label key={value} className={`profile ${mode === value ? 'selected' : ''}`}>
        <input type="radio" name="pdf-scope" checked={mode === value} disabled={disabled || isUnavailable} onChange={() => setMode(value)}/><strong>{label}</strong><small>{isUnavailable ? 'Requires at least 2 original PDFs in your library' : help}</small>
      </label>
    })}</fieldset>
    {mode !== 'current' ? <><fieldset className="row wrap"><legend>Select 2–3 original PDFs</legend>{originals.map(document => <label className="checkbox" key={document.id}>
      <input type="checkbox" checked={selected.includes(document.id)} disabled={disabled || selected.length >= 3 && !selected.includes(document.id)}
        onChange={event => setSelected(event.target.checked ? [...selected, document.id] : selected.filter(id => id !== document.id))}/>{document.name}
    </label>)}</fieldset><p className="muted">{mode === 'separate' ? 'The question count applies to each PDF. Each PDF keeps its own dataset and report.' : 'The question count applies to the combined collection. Citations link to combined and original PDF pages.'} Preparation runs first when needed.</p></> : null}
  </section>
}
