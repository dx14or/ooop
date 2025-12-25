import hashlib
import json
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from .exceptions import CacheError
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CachePaths:
    model_path: Path
    topics_path: Path

@dataclass(frozen=True)
class CacheKey:
    source_size: int
    source_mtime: int
    topic_count: int
    context_size: int
    model_version: int
    auto_stopwords_top_n: int
    proper_noun_min_count: int
    proper_noun_ratio: float
    extra_tag: str = ''

    @classmethod
    def from_source(cls, source: Path, topic_count: int, context_size: int, model_version: int, auto_stopwords_top_n: int, proper_noun_min_count: int, proper_noun_ratio: float, extra_tag: str='') -> 'CacheKey':
        stat = source.stat()
        return cls(source_size=stat.st_size, source_mtime=int(stat.st_mtime), topic_count=topic_count, context_size=context_size, model_version=model_version, auto_stopwords_top_n=auto_stopwords_top_n, proper_noun_min_count=proper_noun_min_count, proper_noun_ratio=proper_noun_ratio, extra_tag=extra_tag)

    @property
    def fingerprint(self) -> str:
        payload = f'{self.source_size}|{self.source_mtime}|{self.topic_count}|{self.context_size}|{self.model_version}|{self.auto_stopwords_top_n}|{self.proper_noun_min_count}|{self.proper_noun_ratio}|{self.extra_tag}'
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

def _safe_stem(path: Path) -> str:
    stem = re.sub('[^A-Za-z0-9_-]+', '_', path.stem)
    return stem or 'dataset'

def _fingerprint(path: Path, topic_count: int, context_size: int, model_version: int, auto_stopwords_top_n: int, proper_noun_min_count: int, proper_noun_ratio: float, extra_tag: str) -> str:
    stat = path.stat()
    payload = f'{stat.st_size}|{int(stat.st_mtime)}|{topic_count}|{context_size}|{model_version}|{auto_stopwords_top_n}|{proper_noun_min_count}|{proper_noun_ratio}|{extra_tag}'
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]

def cache_prefix(base_dir: Path, source: Path, topic_count: int, context_size: int, model_version: int, auto_stopwords_top_n: int, proper_noun_min_count: int, proper_noun_ratio: float, extra_tag: str='') -> str:
    base_dir.mkdir(parents=True, exist_ok=True)
    prefix = '{}_{}'.format(_safe_stem(source), _fingerprint(source, topic_count, context_size, model_version, auto_stopwords_top_n, proper_noun_min_count, proper_noun_ratio, extra_tag))
    return prefix

def cache_paths(base_dir: Path, source: Path, topic_count: int, context_size: int, model_version: int, auto_stopwords_top_n: int, proper_noun_min_count: int, proper_noun_ratio: float, extra_tag: str='') -> CachePaths:
    prefix = cache_prefix(base_dir, source, topic_count, context_size, model_version, auto_stopwords_top_n, proper_noun_min_count, proper_noun_ratio, extra_tag)
    return CachePaths(model_path=base_dir / f'{prefix}_lda.pkl', topics_path=base_dir / f'{prefix}_topics.json')

def load_model(path: Path) -> Any:
    logger.debug('Loading model from %s', path)
    try:
        with open(path, 'rb') as fp:
            return pickle.load(fp)
    except (OSError, pickle.UnpicklingError) as e:
        raise CacheError(f'Failed to load model from {path}: {e}') from e

def save_model(path: Path, model: Any) -> None:
    logger.debug('Saving model to %s', path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as fp:
            pickle.dump(model, fp)
    except (OSError, pickle.PicklingError) as e:
        raise CacheError(f'Failed to save model to {path}: {e}') from e

def load_topics(path: Path) -> dict[str, Any]:
    logger.debug('Loading topics from %s', path)
    try:
        with open(path, 'r', encoding='utf-8') as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError) as e:
        raise CacheError(f'Failed to load topics from {path}: {e}') from e

def save_topics(path: Path, topics: Iterable[int], terms: Iterable[str]) -> None:
    logger.debug('Saving topics to %s', path)
    payload = {'topics': list(topics), 'terms': list(terms)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False)
