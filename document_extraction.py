"""Native PDF extraction with local OCR; no Poppler or Tesseract required by default."""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import shutil
import threading
from dataclasses import dataclass
from html import escape
from pathlib import Path

logger = logging.getLogger(__name__)
_pdfium_lock = threading.Lock()


class ExtractionError(RuntimeError):
    """An actionable document extraction failure."""


@dataclass(frozen=True)
class ExtractionSettings:
    mode: str = "auto"
    ocr_backend: str = "rapidocr"
    dpi: int = 200
    max_pixels: int = 12_000_000
    max_pages: int = 300
    max_file_mb: int = 50
    min_confidence: float = 0.5

    def __post_init__(self):
        if self.mode not in {"auto", "hi_res"}:
            raise ValueError("PDF_EXTRACTION_MODE must be auto or hi_res.")
        if self.ocr_backend not in {"rapidocr", "tesseract"}:
            raise ValueError("OCR_BACKEND must be rapidocr or tesseract.")
        if not 72 <= self.dpi <= 400:
            raise ValueError("OCR_DPI must be between 72 and 400.")
        if min(self.max_pixels, self.max_pages, self.max_file_mb) <= 0:
            raise ValueError("PDF resource limits must be positive.")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("OCR_MIN_CONFIDENCE must be between 0 and 1.")

    @classmethod
    def from_env(cls, *, mode=None, ocr_backend=None):
        return cls(
            mode=mode or os.getenv("PDF_EXTRACTION_MODE", "auto"),
            ocr_backend=ocr_backend or os.getenv("OCR_BACKEND", "rapidocr"),
            dpi=int(os.getenv("OCR_DPI", "200")),
            max_pixels=int(os.getenv("OCR_MAX_PIXELS", "12000000")),
            max_pages=int(os.getenv("PDF_MAX_PAGES", "300")),
            max_file_mb=int(os.getenv("PDF_MAX_FILE_MB", "50")),
            min_confidence=float(os.getenv("OCR_MIN_CONFIDENCE", "0.5")),
        )


def configure_tesseract() -> str:
    """Discover an optional installation without changing PATH or installing tools."""
    configured = os.getenv("TESSERACT_CMD")
    candidates = [configured] if configured else [
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    command = next((str(p) for p in candidates if p and Path(p).is_file()), None)
    if not command:
        raise ExtractionError(
            "Tesseract was not found. Use OCR_BACKEND=rapidocr (the default), "
            "or set TESSERACT_CMD to your installed tesseract executable."
        )
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = command
    try:
        languages = pytesseract.get_languages(config="")
    except Exception as exc:
        raise ExtractionError("Tesseract could not load its language data.") from exc
    if "eng" not in languages:
        raise ExtractionError("Tesseract is missing English (eng) trained data.")
    # Unstructured uses a separate wrapper in its optional high-resolution mode.
    try:
        import unstructured_pytesseract

        unstructured_pytesseract.pytesseract.tesseract_cmd = command
    except ImportError:
        pass
    return command


def recognize_image(image, settings: ExtractionSettings) -> tuple[str, float | None]:
    """Return OCR text and mean accepted-line confidence; confidence is not accuracy."""
    if settings.ocr_backend == "tesseract":
        configure_tesseract()
        import pytesseract

        try:
            return pytesseract.image_to_string(image, lang="eng", timeout=60).strip(), None
        except Exception as exc:
            raise ExtractionError("Tesseract OCR failed for this page.") from exc

    from ocr_runtime import recognize

    return recognize(image, min_confidence=settings.min_confidence)


def _render_page(pdf, page_index: int, settings: ExtractionSettings):
    """Render with PDFium, bounding allocation before creating a bitmap."""
    page = pdf[page_index]
    try:
        width, height = page.get_size()
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ExtractionError("PDF has invalid page dimensions.")
        scale = min(settings.dpi / 72, math.sqrt(settings.max_pixels / (width * height)))
        bitmap = page.render(scale=scale)
        try:
            # PIL must own its pixels before PDFium releases the backing buffer.
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

    def metadata(**kwargs):
        return ElementMetadata(filename=filename, page_number=number, languages=["eng"], **kwargs)

    text = (page.extract_text(layout=False) or "").strip()
    # Short native covers remain native; scan pages with a small text overlay use OCR.
    is_scan = bool(page.images) and len("".join(text.split())) < 80
    if is_scan:
        image = _render_page(pdf, index, settings)
        try:
            ocr_text, confidence = recognize_image(image, settings)
            if not ocr_text:
                # Retain the visual page and flag the result; blank/photo pages may be intentional.
                logger.warning("No OCR text detected on page %s; visual evidence retained.", number)
            elements = []
            if ocr_text or text:
                elements.append(NarrativeText(
                    text=ocr_text or text,
                    metadata=metadata(detection_class_prob=confidence),
                    detection_origin=settings.ocr_backend,
                ))
            elements.append(Image(
                text="",
                metadata=metadata(image_base64=_image_payload(image), image_mime_type="image/jpeg"),
                detection_origin="pdfium_scan",
            ))
            return elements
        finally:
            image.close()

    # Detect native ruled tables and exclude their text from the narrative to avoid duplication.
    tables = page.find_tables()
    narrative_page = page
    if tables:
        boxes = [table.bbox for table in tables]

        def outside_tables(obj):
            x = (obj.get("x0", 0) + obj.get("x1", 0)) / 2
            y = (obj.get("top", 0) + obj.get("bottom", 0)) / 2
            return not any(left <= x <= right and top <= y <= bottom for left, top, right, bottom in boxes)

        narrative_page = page.filter(outside_tables)
        text = (narrative_page.extract_text(layout=False) or "").strip()
    elements = []
    if text:
        elements.append(NarrativeText(text=text, metadata=metadata(), detection_origin="pdfplumber"))
    for table in tables:
        rows = table.extract()
        if rows and any(any(cell for cell in row) for row in rows):
            elements.append(Table(
                text="\n".join("\t".join(str(cell or "") for cell in row) for row in rows),
                metadata=metadata(text_as_html=_table_html(rows)),
                detection_origin="pdfplumber",
            ))
    # A page preview also preserves vector charts that do not appear as embedded images.
    has_vector_graphics = len(page.curves) >= 3 or (not tables and len(page.lines) >= 8)
    if page.images or has_vector_graphics:
        image = _render_page(pdf, index, settings)
        try:
            elements.append(Image(
                text="",
                metadata=metadata(image_base64=_image_payload(image), image_mime_type="image/jpeg"),
                detection_origin="pdfium_visual",
            ))
        finally:
            image.close()
    return elements


def extract_document(file_path, *, settings: ExtractionSettings | None = None,
                     progress_callback=None, cancellation_event=None):
    """Return existing Unstructured-compatible elements with page/source provenance."""
    settings = settings or ExtractionSettings.from_env()
    path = Path(file_path)
    if not path.is_file():
        raise ExtractionError(f"PDF not found: {path.name}")
    if path.stat().st_size > settings.max_file_mb * 1024 * 1024:
        raise ExtractionError(f"PDF exceeds the {settings.max_file_mb} MB file limit.")
    with path.open("rb") as stream:
        if b"%PDF-" not in stream.read(1024):
            raise ExtractionError("File does not contain a PDF header.")
    try:
        import pdfplumber
        import pypdfium2
    except ImportError as exc:
        raise ExtractionError("Install extraction dependencies: python -m pip install -r requirements-ocr.txt") from exc

    try:
        with pdfplumber.open(path) as document:
            if len(document.pages) > settings.max_pages:
                raise ExtractionError(f"PDF exceeds the {settings.max_pages} page limit.")
            if settings.mode == "hi_res":
                if progress_callback:
                    progress_callback(0, 1, "Running high-resolution layout extraction")
                if cancellation_event and cancellation_event.is_set():
                    raise ExtractionError("Document extraction cancelled.")
                return _extract_high_resolution(path, settings)
            # PDFium is not thread-safe; prevent concurrent native rendering in Streamlit.
            with _pdfium_lock, pypdfium2.PdfDocument(str(path)) as pdf:
                elements = []
                total = len(document.pages)
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


def _extract_high_resolution(path: Path, settings: ExtractionSettings):
    """Explicit legacy layout mode, retaining its separately documented dependencies."""
    if settings.ocr_backend != "tesseract":
        raise ExtractionError(
            "Unstructured hi_res requires its supported OCR adapter. Set OCR_BACKEND=tesseract "
            "and install requirements-hires.txt, or use PDF_EXTRACTION_MODE=auto for RapidOCR."
        )
    configure_tesseract()
    if not shutil.which("pdftoppm"):
        raise ExtractionError("Optional hi_res mode requires Poppler (pdftoppm) on PATH.")
    try:
        from unstructured.partition.pdf import partition_pdf

        return partition_pdf(
            filename=str(path), strategy="hi_res", languages=["eng"],
            infer_table_structure=True, extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )
    except Exception as exc:
        raise ExtractionError(f"Optional high-resolution extraction failed: {exc}") from exc
