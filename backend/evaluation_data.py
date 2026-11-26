"""Validated, document-linked evaluation cases and deterministic diagnostic metrics."""

from __future__ import annotations

import csv
import io
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


class EvaluationDataError(ValueError):
    """An evaluation dataset cannot be meaningfully scored against this document."""


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    question: str
    reference_answer: str
    document_hash: str
    answerable: bool = True
    reference_pages: tuple[int, ...] = ()
    relevant_chunk_ids: tuple[str, ...] = ()
    relevance_grades: tuple[tuple[str, float], ...] = ()
    relevant_chunk_groups: tuple[tuple[str, ...], ...] = ()

    def to_dict(self):
        result = asdict(self)
        result["relevance_grades"] = dict(self.relevance_grades)
        return result


def _list_value(value, name):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("["):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise EvaluationDataError(f"{name} must be a JSON array or semicolon-separated values.") from exc
        else:
            value = [item.strip() for item in value.split(";") if item.strip()]
    if not isinstance(value, (list, tuple)):
        raise EvaluationDataError(f"{name} must be a list.")
    return list(value)


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise EvaluationDataError("answerable must be true or false.")


def _case(row, index, shared_hash):
    if isinstance(row, EvaluationCase):
        row = row.to_dict()
    if not isinstance(row, dict):
        raise EvaluationDataError(f"Question {index + 1} must be an object.")
    case_id = str(row.get("id") or f"q{index + 1:04d}").strip()
    question = str(row.get("question") or row.get("user_input") or "").strip()
    reference = str(row.get("reference_answer") or row.get("ground_truth") or row.get("reference") or "").strip()
    document_hash = str(row.get("document_hash") or shared_hash or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{64}", document_hash):
        raise EvaluationDataError(f"{case_id}: document_hash must be the source PDF's SHA-256.")
    if not case_id or len(case_id) > 100:
        raise EvaluationDataError("Question IDs must contain 1-100 characters.")
    if not question or len(question) > 4000:
        raise EvaluationDataError(f"{case_id}: question must contain 1-4000 characters.")
    if not reference or len(reference) > 30000:
        raise EvaluationDataError(f"{case_id}: a reviewed reference_answer is required (maximum 30000 characters).")
    answerable = _boolean(row.get("answerable", True))
    pages = _list_value(row.get("reference_pages"), "reference_pages")
    try:
        if any(isinstance(page, bool) or float(page) != int(page) or int(page) < 1 for page in pages):
            raise ValueError("Invalid page number")
        pages = tuple(sorted({int(page) for page in pages}))
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvaluationDataError(f"{case_id}: reference_pages must contain positive page numbers.") from exc
    ids = _list_value(row.get("relevant_chunk_ids"), "relevant_chunk_ids")
    if any(not isinstance(chunk_id, str) or not chunk_id.strip() for chunk_id in ids):
        raise EvaluationDataError(f"{case_id}: relevant_chunk_ids must contain non-empty strings.")
    ids = tuple(sorted({chunk_id.strip() for chunk_id in ids}))
    groups = []
    for group in _list_value(row.get('relevant_chunk_groups'), 'relevant_chunk_groups'):
        if not isinstance(group, (list, tuple)) or not group or any(not isinstance(item, str) or not item.strip() for item in group):
            raise EvaluationDataError(f'{case_id}: relevant_chunk_groups must contain non-empty lists of chunk IDs.')
        groups.append(tuple(sorted({item.strip() for item in group})))
    if groups and set(ids) != {item for group in groups for item in group}:
        raise EvaluationDataError(f'{case_id}: relevant_chunk_groups must cover exactly the relevant_chunk_ids.')
    grades = row.get("relevance_grades") or {}
    if isinstance(grades, str):
        try:
            grades = json.loads(grades)
        except json.JSONDecodeError as exc:
            raise EvaluationDataError(f"{case_id}: relevance_grades must be a JSON object.") from exc
    if not isinstance(grades, dict):
        raise EvaluationDataError(f"{case_id}: relevance_grades must be an object of chunk ID to relevance.")
    clean_grades = []
    for chunk_id, value in grades.items():
        try:
            if not isinstance(chunk_id, str) or not chunk_id or isinstance(value, bool):
                raise ValueError("Invalid relevance label")
            grade = float(value)
            if not math.isfinite(grade) or not 0 <= grade <= 3:
                raise ValueError("Relevance grade must be between 0 and 3")
        except (TypeError, ValueError) as exc:
            raise EvaluationDataError(f"{case_id}: relevance grades must be finite numbers from 0 to 3.") from exc
        clean_grades.append((chunk_id, grade))
    ids = tuple(sorted(set(ids) | {chunk_id for chunk_id, grade in clean_grades if grade > 0}))
    if groups and set(ids) != {item for group in groups for item in group}:
        raise EvaluationDataError(f'{case_id}: evidence groups must cover all positive relevance labels.')
    if answerable and not (pages or ids):
        raise EvaluationDataError(f"{case_id}: answerable cases need reference_pages or relevant_chunk_ids as reviewed source evidence.")
    if groups and any(grade == 0 and chunk_id in ids for chunk_id, grade in clean_grades):
        raise EvaluationDataError(f'{case_id}: evidence alternatives cannot be labelled as irrelevant.')
    return EvaluationCase(case_id, question, reference, document_hash, answerable, pages, ids, tuple(sorted(clean_grades)), tuple(groups))


def load_evaluation_dataset(data, *, format=None, expected_document_hash=None):
    """Load JSON/CSV bytes, a file path, or already-decoded rows without using guessed references."""
    if data is None:
        raise EvaluationDataError("Import reviewed questions/reference answers for this PDF before starting evaluation.")
    if isinstance(data, Path) or (isinstance(data, str) and format is None and not data.lstrip().startswith(("[", "{"))):
        path = Path(data)
        if not path.is_file():
            raise EvaluationDataError("Evaluation dataset file was not found.")
        format = format or path.suffix.lstrip(".")
        data = path.read_bytes()
    if isinstance(data, bytes):
        if len(data) > 10 * 1024 * 1024:
            raise EvaluationDataError("Evaluation dataset exceeds the 10 MB limit.")
        try:
            data = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise EvaluationDataError("Evaluation datasets must use UTF-8 encoding.") from exc
    if isinstance(data, str):
        format = (format or "json").lower()
        try:
            if format == "csv":
                data = list(csv.DictReader(io.StringIO(data)))
            elif format == "json":
                data = json.loads(data)
            else:
                raise EvaluationDataError("Evaluation datasets must be JSON or CSV.")
        except (csv.Error, json.JSONDecodeError) as exc:
            raise EvaluationDataError(f"Could not parse the {format.upper()} evaluation dataset.") from exc
    shared_hash = None
    if isinstance(data, dict):
        shared_hash = data.get("document_hash")
        data = data.get("questions", data.get("cases"))
    if not isinstance(data, (list, tuple)) or not 1 <= len(data) <= 1000:
        raise EvaluationDataError("Provide a list of 1-1000 reviewed evaluation questions.")
    cases = [_case(row, index, shared_hash) for index, row in enumerate(data)]
    if len({case.id for case in cases}) != len(cases):
        raise EvaluationDataError("Question IDs must be unique; duplicate rows distort averages.")
    hashes = {case.document_hash for case in cases}
    if len(hashes) != 1:
        raise EvaluationDataError("One evaluation run must refer to one PDF document hash.")
    if expected_document_hash and hashes != {str(expected_document_hash).lower()}:
        raise EvaluationDataError("This evaluation dataset belongs to another PDF; its document_hash does not match the selected document.")
    return cases


def dataset_template(document_hash):
    """A blank review template; placeholders intentionally cannot pass validation."""
    return {"document_hash": document_hash, "questions": [{
        "id": "q1", "question": "", "reference_answer": "", "answerable": True,
        "reference_pages": [], "relevant_chunk_ids": [],
    }]}


def document_pages(document):
    metadata = document.metadata or {}
    pages = metadata.get("pages", metadata.get("page_number", metadata.get("page", [])))
    if isinstance(pages, str):
        try:
            pages = json.loads(pages)
        except json.JSONDecodeError:
            pages = pages.split(";")
    if isinstance(pages, (int, float)):
        pages = [pages]
    try:
        return {int(page) for page in pages if int(page) > 0}
    except (TypeError, ValueError):
        return set()


def document_id(document):
    return str((document.metadata or {}).get("chunk_id") or getattr(document, "id", None) or "")


class RelevanceLabels(dict):
    """Reviewed chunk grades with optional equivalent source-passage groups."""
    groups = ()


def resolve_relevance(case, documents):
    by_id = {document_id(document): document for document in documents}
    if "" in by_id:
        raise EvaluationDataError("The index needs stable chunk IDs before retrieval metrics can be scored. Rebuild its index.")
    canonical = {chunk_id: chunk_id for chunk_id in by_id}
    for chunk_id, document in by_id.items():
        for alias in _list_value((document.metadata or {}).get("duplicate_chunk_ids"), "duplicate_chunk_ids"):
            canonical[str(alias)] = chunk_id
    unknown = (set(case.relevant_chunk_ids) | set(dict(case.relevance_grades))) - set(canonical)
    if unknown:
        raise EvaluationDataError(f"{case.id}: relevant_chunk_ids are absent from this index. Review labels after reindexing.")
    raw_grades = {chunk_id: 1.0 for chunk_id in case.relevant_chunk_ids}
    raw_grades.update(dict(case.relevance_grades))
    grades = RelevanceLabels()
    for chunk_id, grade in raw_grades.items():
        primary = canonical[chunk_id]
        grades[primary] = max(grades.get(primary, grade), grade)
    # Page references locate the source, but must not widen reviewed chunk labels.
    # In particular, a grade of zero is an explicit negative label.
    if not (case.relevant_chunk_ids or case.relevance_grades):
        for chunk_id, document in by_id.items():
            if set(case.reference_pages) & document_pages(document):
                grades[chunk_id] = 1.0
    if case.answerable and not any(grade > 0 for grade in grades.values()):
        raise EvaluationDataError(f"{case.id}: no reviewed reference pages/chunks resolve in this index.")
    grades.groups = tuple(frozenset(canonical[item] for item in group) for group in case.relevant_chunk_groups)
    return grades


def retrieval_metrics(retrieved_ids, grades, *, k=5):
    """Binary precision/recall/MRR and graded nDCG over unique stable IDs."""
    relevant = {chunk_id for chunk_id, grade in grades.items() if grade > 0}
    if not relevant:
        return {}
    if not isinstance(k, int) or k < 1:
        raise ValueError("Retrieval k must be positive.")
    ranked = list(dict.fromkeys(retrieved_ids))[:k]
    hits = [chunk_id in relevant for chunk_id in ranked]
    gains = [float(grades.get(chunk_id, 0)) for chunk_id in ranked]
    groups = getattr(grades, 'groups', ())
    if groups:
        grouped_ids = set().union(*groups)
        group_max_grades = [max((float(grades.get(chunk_id, 0)) for chunk_id in group), default=0.0) for group in groups]
        ungrouped_grades = [float(grade) for chunk_id, grade in grades.items() if chunk_id not in grouped_ids]
        ideal_grades = [g for g in group_max_grades + ungrouped_grades if g > 0]
        ideal = sorted(ideal_grades, reverse=True)[:k]
    else:
        ideal = sorted((float(value) for value in grades.values()), reverse=True)[:k]
    dcg = sum((2 ** grade - 1) / math.log2(position + 2) for position, grade in enumerate(gains))
    idcg = sum((2 ** grade - 1) / math.log2(position + 2) for position, grade in enumerate(ideal))
    return {
        "precision_at_k": sum(hits) / k,
        "recall_at_k": sum(bool(group & set(ranked)) for group in grades.groups) / len(grades.groups)
            if getattr(grades, 'groups', ()) else sum(hits) / len(relevant),
        "mrr": next((1 / position for position, hit in enumerate(hits, 1) if hit), 0.0),
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def lexical_reference_metrics(response, reference):
    """Lexical diagnostics only; neither metric establishes factual correctness."""
    def normalized(text):
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    response, reference = normalized(response), normalized(reference)
    token_pattern = r"\d+(?:[.,]\d+)*|[^\W\d_]+"
    actual, expected = Counter(re.findall(token_pattern, response)), Counter(re.findall(token_pattern, reference))
    overlap = sum((actual & expected).values())
    precision = overlap / sum(actual.values()) if actual else 0.0
    recall = overlap / sum(expected.values()) if expected else 0.0
    return {
        "normalized_exact_match": float(response == reference),
        "token_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def validate_score(name, value):
    try:
        if isinstance(value, bool):
            raise ValueError("Boolean is not a metric score")
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: score is not a number.") from exc
    # Ragas answer relevancy uses cosine similarity; a negative cosine is possible.
    minimum = -1 if name == "answer_relevancy" else 0
    if not math.isfinite(value) or not minimum <= value <= 1:
        raise ValueError(f"{name}: score {value} is outside its valid range [{minimum}, 1].")
    return value


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", str(text or "")).split())


def draft_source_texts(documents):
    """Remove recurring page footers from drafting, preserving original indexed evidence."""
    texts = [str(document.page_content or "").strip() for document in documents]
    page_footer = re.compile(r"\s*\|\s*Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)
    prefixes = [
        re.split(r"[.!?]\s+|[\r\n]", text[:match.start()])[-1].split()
        for text in texts if (match := page_footer.search(text))
    ]
    suffixes = Counter(
        " ".join(words[-size:]) for words in prefixes
        for size in range(2, min(12, len(words)) + 1)
    )
    minimum = max(3, math.ceil(len(prefixes) * .2))
    candidates = [suffix for suffix, count in suffixes.items() if count >= minimum]
    labels = [label for label in candidates if not any(other.endswith(" " + label) for other in candidates)]
    patterns = [
        re.compile(r"\s*" + re.escape(label).replace(r"\ ", r"\s+") + page_footer.pattern, re.IGNORECASE)
        for label in labels
    ]
    cleaned = []
    for text in texts:
        for pattern in patterns:
            text = pattern.sub("", text)
        cleaned.append(text.strip())
    return cleaned, labels


def evidence_labels(draft, sources):
    """A complete repeated supporting passage is an alternative, not another required hit."""
    quotes_by_chunk = {}
    for evidence in draft.supporting_quotes:
        quotes_by_chunk.setdefault(evidence.chunk_id, []).append(normalized(evidence.quote))
    groups = []
    stop_words = {
        "a", "an", "the", "what", "which", "who", "how", "when", "where", "why", "is", "are",
        "was", "were", "does", "do", "did", "of", "for", "in", "to", "and", "or", "according",
        "provided", "given", "text", "document", "from", "state", "describe",
    }
    anchors = {
        word.casefold() for word in re.findall(r"\w+", draft.question)
        if word.casefold() not in stop_words or len(word) == 1 and word.isupper()
    }
    for chunk_id, quotes in sorted(quotes_by_chunk.items()):
        quoted_terms = {word.casefold() for quote in quotes for word in re.findall(r"\w+", quote)}
        expand = bool(anchors) and anchors <= quoted_terms
        matches = sorted(
            source["chunk_id"] for source in sources
            if (source["chunk_id"] == chunk_id or expand)
            and all(quote in normalized(source["text"]) for quote in quotes)
        )
        if chunk_id not in matches:
            raise ValueError("Supporting evidence does not occur in its original source chunk.")
        if matches not in groups:
            groups.append(matches)
    return sorted({chunk_id for group in groups for chunk_id in group}), groups
