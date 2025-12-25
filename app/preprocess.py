import re
from collections import Counter
from typing import Iterable
from .lemmatize import lemmatize_text
_URL_PATTERN = re.compile('https?://\\S+|www\\.\\S+')
_TOKEN_RE = re.compile('[A-Za-zА-Яа-яЁё]+')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = _URL_PATTERN.sub(' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub('\\s+', ' ', text)
    return text.strip()

def normalize_rows(rows: Iterable[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for row in rows:
        text = clean_text(row.get('text', ''))
        text = lemmatize_text(text)
        if not text:
            continue
        cleaned.append({**row, 'text': text})
    return cleaned

def normalize_rows_with_raw(rows: Iterable[dict]) -> tuple[list[dict], list[str]]:
    cleaned: list[dict] = []
    raw_texts: list[str] = []
    for row in rows:
        raw = clean_text(row.get('text', ''))
        if not raw:
            continue
        text = lemmatize_text(raw)
        if not text:
            continue
        raw_texts.append(raw)
        cleaned.append({**row, 'text': text})
    return (cleaned, raw_texts)

def top_frequent_terms(texts: Iterable[str], top_n: int) -> set[str]:
    if top_n <= 0:
        return set()
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update((token.lower() for token in _TOKEN_RE.findall(text)))
    return {term for term, _ in counter.most_common(top_n)}

def detect_proper_nouns(texts: Iterable[str], min_count: int, ratio_threshold: float) -> set[str]:
    total: Counter[str] = Counter()
    capitalized: Counter[str] = Counter()
    for text in texts:
        for token in _TOKEN_RE.findall(text):
            lowered = token.lower()
            total[lowered] += 1
            if token[:1].isupper() or token.isupper():
                capitalized[lowered] += 1
    result = set()
    for token, count in total.items():
        if count < min_count:
            continue
        if capitalized[token] / count >= ratio_threshold:
            result.add(token)
    return result
