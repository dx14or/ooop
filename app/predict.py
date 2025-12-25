from collections import Counter, defaultdict
from datetime import datetime
from typing import Iterable

def _time_bucket(dt: datetime) -> str:
    hour = dt.hour
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

def _day_type(dt: datetime) -> str:
    return 'weekend' if dt.weekday() >= 5 else 'weekday'

def build_ngram_model(topics: Iterable[int], context_size: int, timestamps: Iterable[datetime] | None=None) -> dict[tuple[int, ...], Counter]:
    model: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    topics_list = list(topics)
    if timestamps:
        timestamps_list = list(timestamps)
    else:
        timestamps_list = [None] * len(topics_list)
    for idx in range(len(topics_list) - 1):
        next_topic = topics_list[idx + 1]
        max_size = min(context_size, idx + 1)
        for size in range(1, max_size + 1):
            context = tuple(topics_list[idx - size + 1:idx + 1])
            model[context][next_topic] += 1
    return model

def build_ngram_model_with_time(topics: Iterable[int], timestamps: Iterable[datetime], context_size: int) -> dict[tuple, Counter]:
    model: dict[tuple, Counter] = defaultdict(Counter)
    topics_list = list(topics)
    timestamps_list = list(timestamps)
    for idx in range(len(topics_list) - 1):
        next_topic = topics_list[idx + 1]
        next_time_bucket = _time_bucket(timestamps_list[idx + 1])
        next_day_type = _day_type(timestamps_list[idx + 1])
        max_size = min(context_size, idx + 1)
        for size in range(1, max_size + 1):
            context = tuple(topics_list[idx - size + 1:idx + 1])
            time_key = (context, next_time_bucket, next_day_type)
            model[time_key][next_topic] += 1
    return model

def predict_next(model: dict[tuple[int, ...], Counter], recent_topics: Iterable[int], context_size: int, top_k: int, min_count: int, backoff_weights: Iterable[float]) -> list[tuple[int, float]]:
    recent = list(recent_topics)
    if not recent:
        return []
    size_max = min(context_size, len(recent))
    weights = list(backoff_weights)
    if len(weights) < size_max:
        weights = weights + [weights[-1]] * (size_max - len(weights))
    weights = weights[:size_max]
    total_weight = sum(weights) or 1.0
    weights = [w / total_weight for w in weights]
    aggregate: Counter[int] = Counter()
    for idx, size in enumerate(range(size_max, 0, -1)):
        context = tuple(recent[-size:])
        if context not in model:
            continue
        counts = model[context]
        if sum(counts.values()) < min_count:
            continue
        weight = weights[idx]
        for topic, count in counts.items():
            aggregate[topic] += weight * count
    if not aggregate:
        global_counts = Counter()
        for counts in model.values():
            global_counts.update(counts)
        aggregate = global_counts
    total = sum(aggregate.values()) or 1.0
    ranked = aggregate.most_common(top_k)
    return [(topic, count / total) for topic, count in ranked]

def predict_next_with_time(model: dict[tuple, Counter], recent_topics: Iterable[int], current_time: datetime, context_size: int, top_k: int, min_count: int, backoff_weights: Iterable[float]) -> list[tuple[int, float]]:
    recent = list(recent_topics)
    if not recent:
        return []
    time_bucket = _time_bucket(current_time)
    day_type = _day_type(current_time)
    size_max = min(context_size, len(recent))
    weights = list(backoff_weights)
    if len(weights) < size_max:
        weights = weights + [weights[-1]] * (size_max - len(weights))
    weights = weights[:size_max]
    total_weight = sum(weights) or 1.0
    weights = [w / total_weight for w in weights]
    aggregate: Counter[int] = Counter()
    for idx, size in enumerate(range(size_max, 0, -1)):
        context = tuple(recent[-size:])
        time_key = (context, time_bucket, day_type)
        if time_key not in model:
            continue
        counts = model[time_key]
        if sum(counts.values()) < min_count:
            continue
        weight = weights[idx]
        for topic, count in counts.items():
            aggregate[topic] += weight * count
    if not aggregate:
        for idx, size in enumerate(range(size_max, 0, -1)):
            context = tuple(recent[-size:])
            for key, counts in model.items():
                if isinstance(key, tuple) and len(key) >= 1 and key[0] == context:
                    weight = weights[idx]
                    for topic, count in counts.items():
                        aggregate[topic] += weight * count
    if not aggregate:
        global_counts = Counter()
        for counts in model.values():
            global_counts.update(counts)
        aggregate = global_counts
    total = sum(aggregate.values()) or 1.0
    ranked = aggregate.most_common(top_k)
    return [(topic, count / total) for topic, count in ranked]
