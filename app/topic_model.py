from dataclasses import dataclass
from typing import Iterable, Optional
import math
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from .stopwords import get_stop_words

@dataclass
class TopicModel:
    vectorizer: CountVectorizer
    model: LatentDirichletAllocation

def _build_vectorizer(stop_words: set[str]) -> CountVectorizer:
    return CountVectorizer(max_df=0.95, min_df=5, stop_words=list(stop_words), token_pattern='(?u)\\b[а-яёa-z][а-яёa-z]+\\b')

def fit_topic_model(texts: Iterable[str], topic_count: int, extra_stop_words: Optional[set[str]]=None) -> TopicModel:
    stop_words = get_stop_words()
    if extra_stop_words:
        stop_words = stop_words | extra_stop_words
    vectorizer = _build_vectorizer(stop_words)
    doc_term = vectorizer.fit_transform(texts)
    model = LatentDirichletAllocation(n_components=topic_count, random_state=42, learning_method='batch')
    model.fit(doc_term)
    return TopicModel(vectorizer=vectorizer, model=model)

def select_topic_count(texts: Iterable[str], min_topics: int, max_topics: int, step: int, sample_size: int, test_size: float, max_iter: int, improvement_threshold: float, use_coherence: bool, coherence_top_terms: int, coherence_weight: float, extra_stop_words: Optional[set[str]]=None, random_state: int=42) -> tuple[int, dict]:
    texts_list = list(texts)
    if sample_size and len(texts_list) > sample_size:
        texts_list = texts_list[:sample_size]
    stop_words = get_stop_words()
    if extra_stop_words:
        stop_words = stop_words | extra_stop_words
    vectorizer = _build_vectorizer(stop_words)
    doc_term = vectorizer.fit_transform(texts_list)
    vocab_size = doc_term.shape[1]
    if vocab_size < 5 or doc_term.shape[0] < 20:
        fallback = max(2, min_topics)
        return (fallback, {'method': 'fallback', 'reason': 'insufficient_data', 'selected': fallback})
    max_topics = min(max_topics, max(2, vocab_size - 1))
    min_topics = min(min_topics, max_topics)
    if step <= 0:
        step = 1
    train, test = train_test_split(doc_term, test_size=test_size, random_state=random_state)
    doc_term_bin = None
    doc_freq = None
    if use_coherence:
        doc_term_bin = doc_term.copy()
        doc_term_bin.data = np.ones_like(doc_term_bin.data)
        doc_freq = np.asarray(doc_term_bin.sum(axis=0)).ravel()
    results: list[dict] = []
    best_k = min_topics
    best_perplexity = None
    prev_perplexity = None
    for k in range(min_topics, max_topics + 1, step):
        model = LatentDirichletAllocation(n_components=k, random_state=random_state, learning_method='batch', max_iter=max_iter)
        model.fit(train)
        perplexity = model.perplexity(test)
        entry = {'k': k, 'perplexity': perplexity}
        if use_coherence:
            entry['coherence'] = _coherence_umass(model, doc_term_bin, doc_freq, coherence_top_terms)
        results.append(entry)
        if best_perplexity is None or perplexity < best_perplexity:
            best_perplexity = perplexity
            best_k = k
        if not use_coherence and prev_perplexity is not None:
            improvement = (prev_perplexity - perplexity) / max(prev_perplexity, 1e-09)
            if improvement < improvement_threshold:
                return (best_k, {'method': 'perplexity_elbow', 'perplexities': results, 'selected': best_k})
        prev_perplexity = perplexity
    if use_coherence and results:
        perps = [item['perplexity'] for item in results]
        cohs = [item['coherence'] for item in results]
        min_p, max_p = (min(perps), max(perps))
        min_c, max_c = (min(cohs), max(cohs))

        def _norm(value: float, min_v: float, max_v: float) -> float:
            if max_v == min_v:
                return 0.0
            return (value - min_v) / (max_v - min_v)
        best_score = None
        best_k = results[0]['k']
        for item in results:
            perp_norm = _norm(item['perplexity'], min_p, max_p)
            coh_norm = _norm(item['coherence'], min_c, max_c)
            score = (1 - coherence_weight) * (1 - perp_norm) + coherence_weight * coh_norm
            item['score'] = score
            if best_score is None or score > best_score:
                best_score = score
                best_k = item['k']
        return (best_k, {'method': 'combined_score', 'weight_coherence': coherence_weight, 'metrics': results, 'selected': best_k})
    return (best_k, {'method': 'perplexity_min', 'perplexities': results, 'selected': best_k})

def _coherence_umass(model: LatentDirichletAllocation, doc_term_bin, doc_freq: np.ndarray, top_n: int) -> float:
    if doc_term_bin is None or doc_freq is None:
        return 0.0
    total = 0.0
    pair_count = 0
    for topic in model.components_:
        top_ids = topic.argsort()[-top_n:][::-1]
        for i in range(1, len(top_ids)):
            wi = top_ids[i]
            for j in range(i):
                wj = top_ids[j]
                co_occ = float(doc_term_bin[:, wi].multiply(doc_term_bin[:, wj]).sum())
                total += math.log((co_occ + 1.0) / (doc_freq[wj] + 1e-12))
                pair_count += 1
    if pair_count == 0:
        return 0.0
    return total / pair_count

def assign_topics(topic_model: TopicModel, texts: Iterable[str]) -> list[int]:
    doc_term = topic_model.vectorizer.transform(texts)
    topic_probs = topic_model.model.transform(doc_term)
    return topic_probs.argmax(axis=1).tolist()

def top_terms(topic_model: TopicModel, n_terms: int=6) -> list[str]:
    feature_names = topic_model.vectorizer.get_feature_names_out()
    terms: list[str] = []
    for topic in topic_model.model.components_:
        top_ids = topic.argsort()[-n_terms:][::-1]
        terms.append(', '.join((feature_names[i] for i in top_ids)))
    return terms
