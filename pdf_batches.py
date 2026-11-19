"""Combine owned PDFs with an original-to-combined page map."""
import hashlib
import json
import tempfile
from pathlib import Path

MAX_BATCH_PDFS = 3


class BatchStageJob:
    def __init__(self, job, index, count, name, *, stages=1):
        self.parent = job
        self.key = hashlib.sha256(f'{job.key}:{index}'.encode()).hexdigest()[:32]
        self.index, self.count, self.name = index, count, name
        self.stage, self.stages = 0, stages
        self.cancellation_event = job.cancellation_event

    def set_stage(self, stage):
        self.stage = stage

    def progress(self, done, total, message):
        fraction = (self.stage + min(1, max(0, done / max(total, 1)))) / self.stages
        self.parent.progress(self.index * 100 + round(fraction * 100), self.count * 100,
                             f'PDF {self.index + 1}/{self.count}: {self.name} — {message}')

    def check_cancelled(self):
        self.parent.check_cancelled()

    def token(self, text):
        self.parent.token(text)


def merge_pdfs(documents, root, *, max_pages_per_file=300):
    import pypdfium2 as pdfium

    root = Path(root)
    directory = root / 'docs'
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    sources = []
    try:
        with pdfium.PdfDocument.new() as combined:
            offset = 0
            for document in documents:
                try:
                    with pdfium.PdfDocument(root / document['file_path']) as pdf:
                        count = len(pdf)
                        if not 1 <= count <= max_pages_per_file:
                            raise ValueError(f"{document['name']} must contain 1–{max_pages_per_file} pages.")
                        combined.import_pages(pdf)
                except pdfium.PdfiumError as exc:
                    raise ValueError(f"{document['name']} could not be opened as a PDF.") from exc
                sources.append({'document_id': document['id'], 'name': document['name'],
                                'sha256': document['sha256'], 'page_start': offset + 1, 'page_count': count})
                offset += count
            with tempfile.NamedTemporaryFile(dir=directory, suffix='.merge', delete=False) as stream:
                temporary = Path(stream.name)
            combined.save(temporary)
        digest = hashlib.sha256()
        with temporary.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        sha = digest.hexdigest()
        path = directory / (sha + '.pdf')
        if not path.exists():
            temporary.replace(path)
        return {'sha256': sha, 'file_path': str(path.relative_to(root)),
                'size_bytes': path.stat().st_size, 'source_documents': sources}
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def original_pages(pages, sources):
    result = []
    for source in sources:
        matching = sorted(page for page in pages if source['page_start'] <= page < source['page_start'] + source['page_count'])
        if matching:
            result.append({'document_id': source['document_id'], 'name': source['name'],
                           'pages': [page - source['page_start'] + 1 for page in matching],
                           'combined_pages': matching})
    return result


def annotate_chunks(documents, sources):
    from evaluation_data import document_pages

    for document in documents:
        document.metadata['source_pages'] = json.dumps(original_pages(document_pages(document), sources), sort_keys=True)
        document.metadata['source'] = 'Combined PDF'
    return documents
