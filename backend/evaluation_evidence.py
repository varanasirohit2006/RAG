"""Content-only drafting and exact alternative evidence labels."""
import math
import re
import unicodedata
from collections import Counter


def normalized(text):
    return ' '.join(unicodedata.normalize("NFKC", str(text or '')).split())


def draft_source_texts(documents):
    """Remove recurring page footers from drafting, preserving original indexed evidence."""
    texts = [str(document.page_content or '').strip() for document in documents]
    page_footer = re.compile(r'\s*\|\s*Page\s+\d+\s+of\s+\d+\s*$', re.IGNORECASE)
    prefixes = [re.split(r'[.!?]\s+|[\r\n]', text[:match.start()])[-1].split()
                for text in texts if (match := page_footer.search(text))]
    suffixes = Counter(' '.join(words[-size:]) for words in prefixes
                       for size in range(2, min(12, len(words)) + 1))
    minimum = max(3, math.ceil(len(prefixes) * .2))
    candidates = [suffix for suffix, count in suffixes.items() if count >= minimum]
    labels = [label for label in candidates if not any(other.endswith(' ' + label) for other in candidates)]
    patterns = [re.compile(r'\s*' + re.escape(label).replace(r'\ ', r'\s+') + page_footer.pattern,
                           re.IGNORECASE) for label in labels]
    cleaned = []
    for text in texts:
        for pattern in patterns:
            text = pattern.sub('', text)
        cleaned.append(text.strip())
    return cleaned, labels


def evidence_labels(draft, sources):
    """A complete repeated supporting passage is an alternative, not another required hit."""
    quotes_by_chunk = {}
    for evidence in draft.supporting_quotes:
        quotes_by_chunk.setdefault(evidence.chunk_id, []).append(normalized(evidence.quote))
    groups = []
    stop_words = {'a', 'an', 'the', 'what', 'which', 'who', 'how', 'when', 'where', 'why', 'is', 'are',
                  'was', 'were', 'does', 'do', 'did', 'of', 'for', 'in', 'to', 'and', 'or', 'according',
                  'provided', 'given', 'text', 'document', 'from', 'state', 'describe'}
    anchors = {word.casefold() for word in re.findall(r'\w+', draft.question)
               if word.casefold() not in stop_words or len(word) == 1 and word.isupper()}
    for chunk_id, quotes in sorted(quotes_by_chunk.items()):
        quoted_terms = {word.casefold() for quote in quotes for word in re.findall(r'\w+', quote)}
        # Do not widen a generic quote to a different invoice/person/topic whose
        # identifying context lies outside the quote. Semantic alternatives need review.
        expand = bool(anchors) and anchors <= quoted_terms
        matches = sorted(source['chunk_id'] for source in sources
                         if (source['chunk_id'] == chunk_id or expand)
                         and all(quote in normalized(source['text']) for quote in quotes))
        if chunk_id not in matches:
            raise ValueError('Supporting evidence does not occur in its original source chunk.')
        if matches not in groups:
            groups.append(matches)
    return sorted({chunk_id for group in groups for chunk_id in group}), groups
