from datetime import datetime
from pathlib import Path
from .cache import cache_paths, cache_prefix, load_model, load_topics as load_cached_topics, save_model, save_topics
from .classifier import ClassifierConfig, load_classifier, predict_labels, save_classifier, train_classifier
from .config import PATHS, SETTINGS
from .ingest import load_jsonl
from .predict import build_ngram_model_with_time, predict_next_with_time
from .preprocess import detect_proper_nouns, normalize_rows_with_raw, top_frequent_terms
from .stopwords import get_stop_words
from .topic_model import assign_topics, fit_topic_model, select_topic_count, top_terms
from .topics import TopicDefinition, load_topics, topics_signature
from .labeling import bootstrap_keywords, label_by_keywords

def run_from_path(path: Path) -> dict:
    rows = load_jsonl(path)
    rows, raw_texts = normalize_rows_with_raw(rows)
    if not rows:
        return {'predictions': [], 'topic_terms': []}
    texts = [row['text'] for row in rows]
    timestamps = []
    for row in rows:
        if row.get('date'):
            try:
                timestamps.append(datetime.fromisoformat(row['date']))
            except (ValueError, TypeError):
                timestamps.append(datetime.now())
        else:
            timestamps.append(datetime.now())
    auto_stop_words = top_frequent_terms(texts, SETTINGS.auto_stopwords_top_n)
    proper_nouns = detect_proper_nouns(raw_texts, SETTINGS.proper_noun_min_count, SETTINGS.proper_noun_ratio)
    auto_stop_words = auto_stop_words | proper_nouns
    if SETTINGS.use_classifier:
        result = _run_classifier(path=path, texts=texts, timestamps=timestamps, auto_stop_words=auto_stop_words)
        if result:
            return result
    return _run_lda(path=path, texts=texts, timestamps=timestamps, auto_stop_words=auto_stop_words)

def _run_classifier(path: Path, texts: list[str], timestamps: list[datetime], auto_stop_words: set[str]) -> dict | None:
    topics = load_topics()
    topic_ids = [topic.id for topic in topics]
    id_to_index = {topic_id: idx for idx, topic_id in enumerate(topic_ids)}
    titles = [topic.title for topic in topics]
    topic_keywords = set().union(*(topic.keywords for topic in topics))
    seed_labels, seed_stats = label_by_keywords(texts, topics, SETTINGS.label_min_hits, SETTINGS.label_allow_ties)
    labels = seed_labels
    labeling_stats = {'seed': seed_stats}
    if SETTINGS.label_bootstrap_terms > 0 and SETTINGS.label_bootstrap_rounds > 0:
        enriched_topics = topics
        for round_idx in range(SETTINGS.label_bootstrap_rounds):
            extra = bootstrap_keywords(texts, labels, enriched_topics, get_stop_words() | auto_stop_words, SETTINGS.label_bootstrap_terms, SETTINGS.label_bootstrap_min_df, SETTINGS.label_bootstrap_max_df)
            if not extra:
                break
            enriched_topics = [TopicDefinition(id=topic.id, title=topic.title, keywords=topic.keywords | extra.get(topic.id, set())) for topic in enriched_topics]
            labels, stats = label_by_keywords(texts, enriched_topics, SETTINGS.label_min_hits, SETTINGS.label_allow_ties)
            labeling_stats[f'bootstrap_{round_idx + 1}'] = stats
    labeled_texts = [text for text, label in zip(texts, labels) if label]
    labeled_labels = [label for label in labels if label]
    unique_labels = set(labeled_labels)
    if len(labeled_texts) < SETTINGS.classifier_min_docs or len(unique_labels) < SETTINGS.classifier_min_labels:
        return {'predictions': [], 'topic_terms': titles, 'topic_count': len(titles), 'warning': 'Not enough labeled data for classifier; adjust keywords or thresholds.', 'labeling': labeling_stats}
    stop_words = get_stop_words() | auto_stop_words - topic_keywords
    classifier_config = ClassifierConfig(min_df=SETTINGS.classifier_min_df, max_df=SETTINGS.classifier_max_df, ngram_range=SETTINGS.classifier_ngram_range, max_iter=SETTINGS.classifier_max_iter)
    signature = topics_signature()
    prefix = cache_prefix(PATHS.models_dir, path, len(titles), SETTINGS.context_size, SETTINGS.model_version, SETTINGS.auto_stopwords_top_n, SETTINGS.proper_noun_min_count, SETTINGS.proper_noun_ratio, signature)
    clf_path = PATHS.models_dir / f'{prefix}_clf.pkl'
    if clf_path.exists():
        classifier = load_classifier(clf_path)
    else:
        classifier = train_classifier(labeled_texts, labeled_labels, stop_words, classifier_config)
        save_classifier(clf_path, classifier)
    predicted_labels = predict_labels(classifier, texts)
    topics_seq = [id_to_index[label] for label in predicted_labels]
    model = build_ngram_model_with_time(topics_seq, timestamps, SETTINGS.context_size)
    current_time = timestamps[-1] if timestamps else datetime.now()
    predictions = predict_next_with_time(model, topics_seq, current_time, SETTINGS.context_size, SETTINGS.prediction_top_k, SETTINGS.prediction_min_count, SETTINGS.prediction_backoff_weights)
    labeled = [{'topic_id': topic_id, 'label': titles[topic_id], 'prob': prob} for topic_id, prob in predictions]
    return {'predictions': labeled, 'topic_terms': titles, 'topic_count': len(titles), 'labeling': labeling_stats, 'mode': 'classifier', 'topics_seq': topics_seq}

def _run_lda(path: Path, texts: list[str], timestamps: list[datetime], auto_stop_words: set[str]) -> dict:
    topic_count = SETTINGS.topic_count
    selection_info = None
    if not topic_count or topic_count <= 0:
        topic_count, selection_info = select_topic_count(texts, SETTINGS.topic_min, SETTINGS.topic_max, SETTINGS.topic_step, SETTINGS.topic_sample_size, SETTINGS.topic_test_size, SETTINGS.topic_max_iter, SETTINGS.topic_improvement_threshold, SETTINGS.topic_use_coherence, SETTINGS.topic_coherence_top_terms, SETTINGS.topic_coherence_weight, auto_stop_words)
    cache = cache_paths(PATHS.models_dir, path, topic_count, SETTINGS.context_size, SETTINGS.model_version, SETTINGS.auto_stopwords_top_n, SETTINGS.proper_noun_min_count, SETTINGS.proper_noun_ratio)
    topics: list[int]
    terms: list[str]
    if cache.topics_path.exists():
        cached = load_cached_topics(cache.topics_path)
        topics = cached.get('topics', [])
        terms = cached.get('terms', [])
    else:
        if cache.model_path.exists():
            topic_model = load_model(cache.model_path)
        else:
            topic_model = fit_topic_model(texts, topic_count, auto_stop_words)
            save_model(cache.model_path, topic_model)
        topics = assign_topics(topic_model, texts)
        terms = top_terms(topic_model)
        save_topics(cache.topics_path, topics, terms)
    if len(topics) != len(texts) or not terms:
        topic_model = fit_topic_model(texts, topic_count, auto_stop_words)
        topics = assign_topics(topic_model, texts)
        terms = top_terms(topic_model)
        save_model(cache.model_path, topic_model)
        save_topics(cache.topics_path, topics, terms)
    model = build_ngram_model_with_time(topics, timestamps, SETTINGS.context_size)
    current_time = timestamps[-1] if timestamps else datetime.now()
    predictions = predict_next_with_time(model, topics, current_time, SETTINGS.context_size, SETTINGS.prediction_top_k, SETTINGS.prediction_min_count, SETTINGS.prediction_backoff_weights)
    labeled = [{'topic_id': topic_id, 'prob': prob, 'terms': terms[topic_id]} for topic_id, prob in predictions]
    result = {'predictions': labeled, 'topic_terms': terms, 'topic_count': topic_count, 'mode': 'lda', 'topics_seq': topics}
    if selection_info:
        result['selection'] = selection_info
    return result
