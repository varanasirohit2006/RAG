import base64
import io
from types import SimpleNamespace

import pytest
from PIL import Image

import document_extraction as extraction
import ocr_runtime
from ocr_evaluation import score_ocr


def write_native_pdf(path, contents, *, width=300, height=300):
    """Small actual PDFs keep extraction tests independent of a PDF generation library."""
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b"",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    pages = []
    for content in contents:
        page_id, stream_id = len(objects) + 1, len(objects) + 2
        pages.append(f"{page_id} 0 R")
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
                        f"/Resources << /Font << /F1 3 0 R >> >> /Contents {stream_id} 0 R >>").encode())
        data = content.encode("ascii")
        objects.append(f"<< /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(pages)}] /Count {len(pages)} >>".encode()
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(document))
        document.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        document.extend(f"{offset:010} 00000 n \n".encode())
    document.extend((f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
                     f"startxref\n{xref}\n%%EOF\n").encode())
    path.write_bytes(document)
    return path


def test_native_pdf_needs_no_ocr_or_external_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(extraction.shutil, "which", lambda name: None)
    def unexpected_ocr(*args, **kwargs):
        pytest.fail("Native text must bypass OCR")
    monkeypatch.setattr(extraction, "recognize_image", unexpected_ocr)
    path = write_native_pdf(tmp_path / "native.pdf", [
        "BT /F1 12 Tf 20 260 Td (Native report total 12345.67 USD.) Tj ET",
        "BT /F1 12 Tf 20 260 Td (Second page evidence.) Tj ET",
    ])
    elements = extraction.extract_document(path, settings=extraction.ExtractionSettings())
    assert [element.metadata.page_number for element in elements] == [1, 2]
    assert "12345.67" in elements[0].text
    assert all(element.category == "NarrativeText" for element in elements)
    assert all(element.metadata.filename == "native.pdf" for element in elements)


def test_native_table_preserves_cells_and_escapes_html(tmp_path):
    content = "\n".join([
        "20 100 m 280 100 l S", "20 150 m 280 150 l S", "20 200 m 280 200 l S",
        "20 100 m 20 200 l S", "150 100 m 150 200 l S", "280 100 m 280 200 l S",
        "BT /F1 12 Tf 30 175 Td (Item) Tj ET",
        "BT /F1 12 Tf 160 175 Td (Value) Tj ET",
        "BT /F1 12 Tf 30 125 Td (a < b) Tj ET",
        "BT /F1 12 Tf 160 125 Td (123.45) Tj ET",
    ])
    path = write_native_pdf(tmp_path / "table.pdf", [content])
    elements = extraction.extract_document(path, settings=extraction.ExtractionSettings())
    tables = [element for element in elements if element.category == "Table"]
    assert len(tables) == 1
    assert "123.45" in tables[0].text
    assert "a &lt; b" in tables[0].metadata.text_as_html
    assert not any("123.45" in element.text for element in elements if element.category == "NarrativeText")


def test_scanned_page_uses_ocr_and_retains_bounded_visual(tmp_path, monkeypatch):
    path = tmp_path / "scan.pdf"
    with Image.new("RGB", (1600, 600), "white") as image:
        image.save(path, "PDF", resolution=200)
    calls = []
    def recognize(image, settings):
        calls.append(image.size)
        return "Scanned invoice total 12345.67 USD.", 0.95
    monkeypatch.setattr(extraction, "recognize_image", recognize)
    settings = extraction.ExtractionSettings(max_pixels=500_000)
    elements = extraction.extract_document(path, settings=settings)
    assert len(calls) == 1
    assert calls[0][0] * calls[0][1] <= settings.max_pixels + 2000
    assert elements[0].text == "Scanned invoice total 12345.67 USD."
    assert elements[0].metadata.detection_class_prob == 0.95
    assert elements[1].category == "Image"
    with Image.open(io.BytesIO(base64.b64decode(elements[1].metadata.image_base64))) as visual:
        assert visual.format == "JPEG"


def test_native_vector_chart_keeps_visual_evidence(tmp_path):
    content = "BT /F1 12 Tf 20 270 Td (Sales chart.) Tj ET\n" + "\n".join(
        f"{20 + i * 20} 40 m {20 + i * 20} {70 + i * 10} l S" for i in range(9)
    )
    path = write_native_pdf(tmp_path / "chart.pdf", [content])
    elements = extraction.extract_document(path, settings=extraction.ExtractionSettings())
    assert any(element.category == "Image" for element in elements)
    assert any("Sales chart" in element.text for element in elements)


def test_mixed_pdf_preserves_pages_and_chunk_visuals(tmp_path, monkeypatch):
    import pypdfium2
    from rag_pipeline import create_chunks_by_title, separate_content_types

    native = write_native_pdf(tmp_path / "native.pdf", [
        "BT /F1 12 Tf 20 260 Td (Native page total 100 USD.) Tj ET",
    ])
    scanned = tmp_path / "scan.pdf"
    with Image.new("RGB", (800, 300), "white") as image:
        image.save(scanned, "PDF", resolution=200)
    mixed = tmp_path / "mixed.pdf"
    with pypdfium2.PdfDocument.new() as target:
        with pypdfium2.PdfDocument(str(native)) as source:
            target.import_pages(source)
        with pypdfium2.PdfDocument(str(scanned)) as source:
            target.import_pages(source)
        target.save(str(mixed))
    calls = []
    def recognize(image, settings):
        calls.append(1)
        return "Scanned second page total 200 USD.", 0.95
    monkeypatch.setattr(extraction, "recognize_image", recognize)
    elements = extraction.extract_document(mixed, settings=extraction.ExtractionSettings())
    assert len(calls) == 1
    assert {element.metadata.page_number for element in elements} == {1, 2}
    chunks = create_chunks_by_title(elements)
    contents = [separate_content_types(chunk) for chunk in chunks]
    assert any("100 USD" in content["text"] for content in contents)
    assert any("200 USD" in content["text"] for content in contents)
    assert any(content["images"] for content in contents)


def test_page_limit_rejects_document_before_processing(tmp_path, monkeypatch):
    path = write_native_pdf(tmp_path / "large.pdf", ["", ""])
    monkeypatch.setattr(extraction, "_page_elements", lambda *args: pytest.fail("Must reject before processing"))
    with pytest.raises(extraction.ExtractionError, match="page limit"):
        extraction.extract_document(path, settings=extraction.ExtractionSettings(max_pages=1))


def test_invalid_file_is_rejected(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_text("This is not a PDF.")
    with pytest.raises(extraction.ExtractionError, match="PDF header"):
        extraction.extract_document(path)


@pytest.mark.parametrize("settings", [
    {"mode": "unknown"}, {"ocr_backend": "unknown"}, {"dpi": 1000},
    {"max_pixels": 0}, {"min_confidence": float("nan")},
])
def test_invalid_configuration_fails_early(settings):
    with pytest.raises(ValueError):
        extraction.ExtractionSettings(**settings)


def test_hi_res_requires_explicit_supported_backend(tmp_path):
    path = write_native_pdf(tmp_path / "native.pdf", ["BT /F1 12 Tf 20 260 Td (Hello.) Tj ET"])
    with pytest.raises(extraction.ExtractionError, match="hi_res"):
        extraction.extract_document(path, settings=extraction.ExtractionSettings(mode="hi_res"))


def test_rapidocr_filters_low_confidence_and_reports_mean(monkeypatch):
    fake = SimpleNamespace(txts=("Invoice", "noise", "123.45"), scores=(0.9, 0.1, 0.8))
    monkeypatch.setattr(ocr_runtime, "get_engine", lambda: lambda image: fake)
    text, confidence = ocr_runtime.recognize(None, min_confidence=0.5)
    assert text == "Invoice\n123.45"
    assert confidence == pytest.approx(0.85)


@pytest.mark.parametrize("scores", [(5.001232,), (float("nan"),), ()])
def test_invalid_ocr_result_is_not_reported_as_success(monkeypatch, scores):
    fake = SimpleNamespace(txts=("Invoice",), scores=scores)
    monkeypatch.setattr(ocr_runtime, "get_engine", lambda: lambda image: fake)
    with pytest.raises(extraction.ExtractionError):
        ocr_runtime.recognize(None)


def test_ocr_error_rates_use_reference_and_preserve_numeric_errors():
    perfect = score_ocr("Total 123.45 USD", "Total\n123.45   USD")
    assert perfect["character_error_rate"] == 0
    assert perfect["word_error_rate"] == 0
    wrong = score_ocr("Total 123.45 USD", "Total 128.45 USD")
    assert wrong["character_error_rate"] == pytest.approx(1 / 16)
    assert wrong["word_error_rate"] == pytest.approx(1 / 3)
    assert score_ocr("a", "a b c")["word_error_rate"] == 2
    with pytest.raises(ValueError, match="reference"):
        score_ocr("", "No ground truth")
