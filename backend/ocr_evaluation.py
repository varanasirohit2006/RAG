"""Deterministic OCR accuracy measures against a verified transcription."""

import re
import unicodedata


def _edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for row, expected in enumerate(reference, 1):
        current = [row]
        for column, actual in enumerate(hypothesis, 1):
            current.append(min(
                previous[column] + 1,
                current[column - 1] + 1,
                previous[column - 1] + (expected != actual),
            ))
        previous = current
    return previous[-1]


def normalize_text(text: str) -> str:
    """Normalize Unicode/whitespace; retain case, numbers, and punctuation."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def score_ocr(reference: str, hypothesis: str) -> dict:
    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    if not reference:
        raise ValueError("OCR accuracy requires a non-empty verified reference transcription.")
    expected_words, actual_words = reference.split(), hypothesis.split()
    # Insertions can make CER/WER exceed 1; these error rates must not be clamped.
    return {
        "character_error_rate": _edit_distance(reference, hypothesis) / len(reference),
        "word_error_rate": _edit_distance(expected_words, actual_words) / len(expected_words),
        "reference_characters": len(reference),
        "reference_words": len(expected_words),
        "normalization": "Unicode NFKC and whitespace; case and punctuation preserved",
    }
