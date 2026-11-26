"""Native PDF text/table extraction with bundled local RapidOCR and multi-PDF batching."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass
from html import escape
from pathlib import Path

logger = logging.getLogger(__name__)
_pdfium_lock = threading.Lock()
_engine_lock = threading.Lock()
MAX_BATCH_PDFS = 3


class ExtractionError(RuntimeError):
    """An actionable document extraction failure."""


@dataclass(frozen=True)
class ExtractionSettings:
    dpi: int = 200
    max_pixels: int = 12_000_000
    max_pages: int = 300
    max_file_mb: int = 50
    min_confidence: float = 0.5

    def __post_init__(self):
        if not 72 <= self.dpi <= 400:
            raise ValueError("OCR_DPI must be between 72 and 400.")
        if min(self.max_pixels, self.max_pages, self.max_file_mb) <= 0:
            raise ValueError("PDF resource limits must be positive.")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("OCR_MIN_CONFIDENCE must be between 0 and 1.")

    @classmethod
    def from_env(cls, **_kwargs):
        return cls(
            dpi=int(os.getenv("OCR_DPI", "200")),
            max_pixels=int(os.getenv("OCR_MAX_PIXELS", "12000000")),
            max_pages=int(os.getenv("PDF_MAX_PAGES", "300")),
            max_file_mb=int(os.getenv("PDF_MAX_FILE_MB", "50")),
            min_confidence=float(os.getenv("OCR_MIN_CONFIDENCE", "0.5")),
        )


def get_engine():
    try:
        import rapidocr
    except ImportError as exc:
        raise ExtractionError("RapidOCR is missing. Run: python -m pip install -r requirements.txt") from exc
    model_dir = Path(rapidocr.__file__).resolve().parent / "models"
    models = {
        "Det.model_path": model_dir / "PP-OCRv6_det_small.onnx",
        "Rec.model_path": model_dir / "PP-OCRv6_rec_small.onnx",
        "Cls.model_path": model_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    }
    if any(not path.is_file() for path in models.values()):
        raise ExtractionError("RapidOCR bundled models are missing. Reinstall rapidocr==3.10.0.")
    threads = int(os.getenv("OCR_THREADS", "2"))
    return rapidocr.RapidOCR(params={
        **{key: str(path) for key, path in models.items()},
        "Det.lang_type": "en",
        "Rec.lang_type": "en",
        "Global.text_score": 0.0,
        "Global.log_level": "warning",
        "EngineConfig.onnxruntime.intra_op_num_threads": threads,
        "EngineConfig.onnxruntime.inter_op_num_threads": 1,
    })


def recognize(image, *, min_confidence: float = 0.5) -> tuple[str, float | None]:
    if not 0 <= min_confidence <= 1:
        raise ValueError("Minimum OCR confidence must be between 0 and 1.")
    with _engine_lock:
        try:
            result = get_engine()(image)
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(f"RapidOCR could not read this image: {exc}") from exc
    texts = result.txts if result.txts is not None else ()
    scores = result.scores if result.scores is not None else ()
    if len(texts) != len(scores):
        raise ExtractionError("RapidOCR returned inconsistent text/confidence counts.")
    accepted = [(t.strip(), float(s)) for t, s in zip(texts, scores) if t.strip() and 0 <= float(s) <= 1 and float(s) >= min_confidence]
    if not accepted:
        return "", None
    return "\n".join(text for text, _ in accepted), sum(s for _, s in accepted) / len(accepted)


def recognize_image(image, settings: ExtractionSettings) -> tuple[str, float | None]:
    return recognize(image, min_confidence=settings.min_confidence)


def _render_page(pdf, page_index: int, settings: ExtractionSettings):
    page = pdf[page_index]
    try:
        width, height = page.get_size()
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ExtractionError("PDF has invalid page dimensions.")
        scale = min(settings.dpi / 72, math.sqrt(settings.max_pixels / (width * height)))
        bitmap = page.render(scale=scale)
        try:
            return bitmap.to_pil().convert("RGB").copy()
        finally:
            bitmap.close()
    finally:
        page.close()


def _image_payload(image) -> str:
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("ascii")


def _table_html(rows) -> str:
    return "<table>" + "".join(
        "<tr>" + "".join(f"<td>{escape(str(cell or ''))}</td>" for cell in row) + "</tr>"
        for row in rows
    ) + "</table>"


def _page_elements(page, pdf, index: int, filename: str, settings: ExtractionSettings):
    from unstructured.documents.elements import ElementMetadata, Image, NarrativeText, Table

    number = index + 1
    meta = lambda **kw: ElementMetadata(filename=filename, page_number=number, languages=["eng"], **kw)
    text = (page.extract_text(layout=False) or "").strip()

    # Run RapidOCR on scanned pages (pages with embedded images and < 80 non-space chars).
    if bool(page.images) and len("".join(text.split())) < 80:
        image = _render_page(pdf, index, settings)
        try:
            ocr_text, confidence = recognize_image(image, settings)
            if not ocr_text:
                logger.warning("No OCR text detected on page %s; visual evidence retained.", number)
            elements = []
            if ocr_text or text:
                elements.append(NarrativeText(text=ocr_text or text, metadata=meta(detection_class_prob=confidence), detection_origin="rapidocr"))
            elements.append(Image(text="", metadata=meta(image_base64=_image_payload(image), image_mime_type="image/jpeg"), detection_origin="pdfium_scan"))
            return elements
        finally:
            image.close()

    # Extract ruled tables separately so table text is not duplicated in narrative chunks.
    tables = page.find_tables()
    if tables:
        boxes = [table.bbox for table in tables]
        narrative_page = page.filter(lambda obj: not any(
            l <= (obj.get("x0", 0) + obj.get("x1", 0)) / 2 <= r and t <= (obj.get("top", 0) + obj.get("bottom", 0)) / 2 <= b
            for l, t, r, b in boxes
        ))
        text = (narrative_page.extract_text(layout=False) or "").strip()

    elements = []
    if text:
        elements.append(NarrativeText(text=text, metadata=meta(), detection_origin="pdfplumber"))
    for table in tables:
        rows = table.extract()
        if rows and any(any(cell for cell in row) for row in rows):
            elements.append(Table(
                text="\n".join("\t".join(str(cell or "") for cell in row) for row in rows),
                metadata=meta(text_as_html=_table_html(rows)),
                detection_origin="pdfplumber",
            ))
    if page.images or len(page.curves) >= 3 or (not tables and len(page.lines) >= 8):
        image = _render_page(pdf, index, settings)
        try:
            elements.append(Image(text="", metadata=meta(image_base64=_image_payload(image), image_mime_type="image/jpeg"), detection_origin="pdfium_visual"))
        finally:
            image.close()
    return elements


def extract_document(file_path, *, settings: ExtractionSettings | None = None,
                     progress_callback=None, cancellation_event=None):
    """Extract text, tables, and visual elements from a PDF without external binaries."""
    settings = settings or ExtractionSettings.from_env()
    path = Path(file_path)
    if not path.is_file():
        raise ExtractionError(f"PDF not found: {path.name}")
    if path.stat().st_size > settings.max_file_mb * 1024 * 1024:
        raise ExtractionError(f"PDF exceeds the {settings.max_file_mb} MB file limit.")
    with path.open("rb") as stream:
        if b"%PDF-" not in stream.read(1024):
            raise ExtractionError("File does not contain a PDF header.")

    import pdfplumber
    import pypdfium2

    try:
        with pdfplumber.open(path) as document:
            if len(document.pages) > settings.max_pages:
                raise ExtractionError(f"PDF exceeds the {settings.max_pages} page limit.")
            with _pdfium_lock, pypdfium2.PdfDocument(str(path)) as pdf:
                elements, total = [], len(document.pages)
                for index, page in enumerate(document.pages):
                    try:
                        if cancellation_event and cancellation_event.is_set():
                            raise ExtractionError("Document extraction cancelled.")
                        if progress_callback:
                            progress_callback(index, total, f"Extracting page {index + 1} of {total}")
                        elements.extend(_page_elements(page, pdf, index, path.name, settings))
                        if progress_callback:
                            progress_callback(index + 1, total, f"Extracted page {index + 1} of {total}")
                    except ExtractionError:
                        raise
                    except Exception as exc:
                        raise ExtractionError(f"Could not extract PDF page {index + 1}: {exc}") from exc
                    finally:
                        page.close()
        if not elements:
            raise ExtractionError("PDF contains no extractable text or visual evidence.")
        return elements
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF; it may be corrupt or password-protected: {exc}") from exc


class BatchStageJob:
    def __init__(self, job, index, count, name, *, stages=1):
        self.parent = job
        self.key = hashlib.sha256(f"{job.key}:{index}".encode()).hexdigest()[:32]
        self.index, self.count, self.name = index, count, name
        self.stage, self.stages = 0, stages
        self.cancellation_event = job.cancellation_event

    def set_stage(self, stage):
        self.stage = stage

    def progress(self, done, total, message):
        fraction = (self.stage + min(1, max(0, done / max(total, 1)))) / self.stages
        self.parent.progress(
            self.index * 100 + round(fraction * 100),
            self.count * 100,
            f"PDF {self.index + 1}/{self.count}: {self.name} — {message}",
        )

    def check_cancelled(self):
        self.parent.check_cancelled()

    def token(self, text):
        self.parent.token(text)


def merge_pdfs(documents, root, *, max_pages_per_file=300):
    import pypdfium2 as pdfium

    root = Path(root)
    directory = root / "docs"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    sources = []
    try:
        with pdfium.PdfDocument.new() as combined:
            offset = 0
            for document in documents:
                try:
                    with pdfium.PdfDocument(root / document["file_path"]) as pdf:
                        count = len(pdf)
                        if not 1 <= count <= max_pages_per_file:
                            raise ValueError(f"{document['name']} must contain 1–{max_pages_per_file} pages.")
                        combined.import_pages(pdf)
                except pdfium.PdfiumError as exc:
                    raise ValueError(f"{document['name']} could not be opened as a PDF.") from exc
                sources.append({
                    "document_id": document["id"], "name": document["name"],
                    "sha256": document["sha256"], "page_start": offset + 1, "page_count": count,
                })
                offset += count
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".merge", delete=False) as stream:
                temporary = Path(stream.name)
            combined.save(temporary)
        sha = hashlib.sha256(temporary.read_bytes()).hexdigest()
        path = directory / (sha + ".pdf")
        if not path.exists():
            temporary.replace(path)
        return {
            "sha256": sha, "file_path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size, "source_documents": sources,
        }
    finally:
        if temporary and temporary.exists():
            temporary.unlink()


def original_pages(pages, sources):
    result = []
    for source in sources:
        matching = sorted(page for page in pages if source["page_start"] <= page < source["page_start"] + source["page_count"])
        if matching:
            result.append({
                "document_id": source["document_id"], "name": source["name"],
                "pages": [page - source["page_start"] + 1 for page in matching],
                "combined_pages": matching,
            })
    return result


def annotate_chunks(documents, sources):
    from backend.evaluation_data import document_pages

    for document in documents:
        document.metadata["source_pages"] = json.dumps(original_pages(document_pages(document), sources), sort_keys=True)
        document.metadata["source"] = "Combined PDF"
    return documents
