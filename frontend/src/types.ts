export interface Config { generation_model: string; evaluation_model: string; max_file_mb: number; reranker_configured: boolean }
export interface DocumentInfo { id: string; name: string; sha256: string; status: string; size_bytes: number; stats: Record<string, number>; timings?: Record<string, number>; chunk_count?: number; extraction_cache_hit?: boolean }
export interface Source { id: string; text: string; pages: number[]; tables: string[]; images: string[]; relevance_score?: number | null }
export interface Answer { id: string; query: string; answer: string; sources: Source[]; timings: Record<string, number | string>; created_at?: string }
export interface EvaluationCase { id: string; question: string; reference_answer: string; reference_pages: number[]; answerable: boolean; relevant_chunk_ids?: string[]; category?: string; supporting_quotes?: (string | { quote?: string; text?: string; chunk_id?: string })[]; verification?: Record<string, unknown> }
export interface Dataset { id: string; name: string; count: number; cases: EvaluationCase[]; origin?: string; provenance?: Record<string, unknown>; warnings?: string[] }
export interface Run { id: string; created_at: string; manifest: Record<string, unknown>; summary: Record<string, string | number | null>[]; cache_reused?: boolean }
export interface RetrievalResult { query: string; queries: string[]; sources: Source[]; per_query: { query: string; sources: Source[] }[]; answer?: string; warning?: string; timings: Record<string, number> }
export interface JobInfo { id: string; document_id: string; kind: string }
export interface Job<T = unknown> extends JobInfo { status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'; done: number; total: number; message: string; elapsed_seconds: number; partial_text: string; events: [number, string][]; error?: string; result?: T }
export const isActive = (job?: Job | null) => !!job && (job.status === 'queued' || job.status === 'running')
