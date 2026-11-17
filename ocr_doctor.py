"""Check OCR tooling, or run a known English scan without credentials or cloud calls."""

import argparse
import importlib.metadata
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

from document_extraction import ExtractionSettings, extract_document, recognize_image
from ocr_evaluation import score_ocr

REFERENCE = "Reliable OCR reads English scans.\nInvoice total is 12345.67 USD."


def make_sample_image():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1600, 500), "white")
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "Arial.ttf",
    ]
    font = None
    for candidate in candidates:
        try:
            font = ImageFont.truetype(candidate, 48)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default(size=48)
    ImageDraw.Draw(image).multiline_text((70, 90), REFERENCE, fill="black", font=font, spacing=35)
    return image


def diagnose():
    packages = {}
    for package in ("rapidocr", "onnxruntime", "pdfplumber", "pypdfium2", "unstructured"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": packages,
        "backend": ExtractionSettings.from_env().ocr_backend,
        "tesseract_on_path": bool(shutil.which("tesseract")),
        "poppler_on_path": bool(shutil.which("pdftoppm")),
        "default_requires_external_executables": False,
        "missing_packages": [package for package, version in packages.items() if not version],
    }


def self_test():
    settings = ExtractionSettings.from_env(mode="auto")
    started = time.perf_counter()
    with make_sample_image() as image:
        text, confidence = recognize_image(image, settings)
        image_seconds = time.perf_counter() - started
        pdf_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="rag-ocr-") as directory:
            path = Path(directory) / "known-english-scan.pdf"
            image.save(path, "PDF", resolution=200)
            elements = extract_document(path, settings=settings)
        pdf_seconds = time.perf_counter() - pdf_started
    extracted = "\n".join(element.text for element in elements if element.category != "Image")
    image_metrics = score_ocr(REFERENCE, text)
    pdf_metrics = score_ocr(REFERENCE, extracted)
    passed = all(
        metrics["character_error_rate"] <= 0.1 and metrics["word_error_rate"] <= 0.25
        for metrics in (image_metrics, pdf_metrics)
    ) and any(element.category == "Image" for element in elements)
    return {
        "passed": passed,
        "reference": REFERENCE,
        "recognized_image_text": text,
        "recognized_pdf_text": extracted,
        "mean_line_confidence": confidence,
        "image_metrics": image_metrics,
        "pdf_metrics": pdf_metrics,
        "seconds": round(time.perf_counter() - started, 3),
        "cold_image_seconds": round(image_seconds, 3),
        "warm_pdf_seconds": round(pdf_seconds, 3),
        "scope": "Synthetic clean English scan; does not establish real-document accuracy.",
    }


def main():
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run image and scanned-PDF OCR.")
    parser.add_argument("--pdf", type=Path, help="Extract a supplied PDF without invoking an LLM.")
    parser.add_argument("--reference", type=Path, help="Verified UTF-8 transcription for --pdf.")
    parser.add_argument("--output", type=Path, help="Save the JSON diagnostic report.")
    args = parser.parse_args()
    if args.reference and not args.pdf:
        parser.error("--reference requires --pdf")
    report = diagnose()
    try:
        if args.self_test:
            report["self_test"] = self_test()
        if args.pdf:
            started = time.perf_counter()
            elements = extract_document(args.pdf)
            text = "\n".join(element.text for element in elements if element.category != "Image")
            report["pdf"] = {
                "filename": args.pdf.name, "text": text, "elements": len(elements),
                "pages": sorted({element.metadata.page_number for element in elements}),
                "seconds": round(time.perf_counter() - started, 3),
            }
            if args.reference:
                report["pdf"]["metrics"] = score_ocr(args.reference.read_text(encoding="utf-8"), text)
    except Exception as exc:
        report["error"] = str(exc)
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    return int(bool(report.get("error")) or bool(report["missing_packages"]) or
               (args.self_test and not report.get("self_test", {}).get("passed", False)))


if __name__ == "__main__":
    raise SystemExit(main())
