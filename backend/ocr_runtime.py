"""A lazily loaded, CPU-only RapidOCR engine using models shipped in its wheel."""

from __future__ import annotations

import math
import os
import threading
from functools import lru_cache
from pathlib import Path

from backend.document_extraction import ExtractionError

_engine_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_engine():
    try:
        import rapidocr
    except ImportError as exc:
        raise ExtractionError(
            "RapidOCR is missing from this Python environment. Activate .venv and run "
            "python -m pip install -r requirements-ocr.txt. No Tesseract installer is needed."
        ) from exc
    model_dir = Path(rapidocr.__file__).resolve().parent / "models"
    models = {
        "Det.model_path": model_dir / "PP-OCRv6_det_small.onnx",
        "Rec.model_path": model_dir / "PP-OCRv6_rec_small.onnx",
        "Cls.model_path": model_dir / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
    }
    if any(not path.is_file() for path in models.values()):
        raise ExtractionError(
            "RapidOCR's bundled models are missing. Reinstall rapidocr==3.10.0 "
            "in the active environment; the app does not download models during OCR."
        )
    try:
        threads = int(os.getenv("OCR_THREADS", "2"))
        if not 1 <= threads <= 16:
            raise ValueError("OCR_THREADS must be between 1 and 16.")
        return rapidocr.RapidOCR(params={
            **{key: str(path) for key, path in models.items()},
            "Det.lang_type": "en",
            "Rec.lang_type": "en",
            "Global.text_score": 0.0,
            "Global.log_level": "warning",
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        })
    except Exception as exc:
        raise ExtractionError(f"Could not initialize RapidOCR: {exc}") from exc


def recognize(image, *, min_confidence: float = 0.5) -> tuple[str, float | None]:
    if not 0 <= min_confidence <= 1:
        raise ValueError("Minimum OCR confidence must be between 0 and 1.")
    # RapidOCR mutates call options internally; serialize use of the shared engine.
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
    accepted = []
    for text, score in zip(texts, scores):
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ExtractionError("RapidOCR returned an invalid confidence score.")
        if text.strip() and score >= min_confidence:
            accepted.append((text.strip(), score))
    if not accepted:
        return "", None
    return "\n".join(text for text, _ in accepted), sum(s for _, s in accepted) / len(accepted)
