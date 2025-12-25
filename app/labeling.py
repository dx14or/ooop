from collections import Counter
from typing import Iterable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .topics import TopicDefinition

def label_by_keywords(texts: Iterable[str], topics: Iterable[TopicDefinition], min_hits: int, allow_ties: bool) -> tuple[list[str | None], dict]:
    labels: list[str | None] = []
    topic_list = list(topics)
    label_counts: Counter[str] = Counter()
    tie_count = 0
    for text in texts:
        tokens = Counter(text.split())
        best_label = None
        best_score = 0
        tie = False
        for topic in topic_list:
            score = sum((tokens.get(word, 0) for word in topic.keywords))
            if score > best_score:
                best_score = score
                best_label = topic.id
                tie = False
            elif score == best_score and score > 0:
                tie = True
        if best_label and best_score >= min_hits and (allow_ties or not tie):
            if tie:
                tie_count += 1
            labels.append(best_label)
            label_counts[best_label] += 1
        else:
            labels.append(None)
    labeled_total = sum(label_counts.values())
    stats = {'labeled': labeled_total, 'total': len(labels), 'coverage': labeled_total / max(len(labels), 1), 'label_counts': dict(label_counts), 'ties': tie_count}
    return (labels, stats)

def bootstrap_keywords(texts: Iterable[str], labels: Iterable[str | None], topics: Iterable[TopicDefinition], stop_words: set[str], top_n: int, min_df: int, max_df: float) -> dict[str, set[str]]:
    topic_texts: dict[str, list[str]] = {}
    for text, label in zip(texts, labels):
        if not label:
            continue
        topic_texts.setdefault(label, []).append(text)
    extra: dict[str, set[str]] = {}
    for topic in topics:
        docs = topic_texts.get(topic.id, [])
        if len(docs) < max(min_df, 3):
            continue
        vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, token_pattern='(?u)\\b[а-яёa-z][а-яёa-z]+\\b', stop_words=list(stop_words))
        try:
            matrix = vectorizer.fit_transform(docs)
        except ValueError as exc:
            msg = str(exc)
            if 'After pruning, no terms remain' in msg or 'empty vocabulary' in msg:
                continue
            raise
        if matrix.shape[1] == 0:
            continue
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()
        top_ids = scores.argsort()[-top_n:][::-1]
        terms = [features[idx] for idx in top_ids if features[idx] not in stop_words and features[idx] not in topic.keywords]
        if terms:
            extra[topic.id] = set(terms)
    return extra
