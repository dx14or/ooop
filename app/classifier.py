from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .text.patterns import SKLEARN_TOKEN_PATTERN
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ClassifierConfig:
    min_df: int
    max_df: float
    ngram_range: tuple[int, int]
    max_iter: int

def train_classifier(texts: Sequence[str], labels: Sequence[str], stop_words: set[str], config: ClassifierConfig) -> Pipeline:
    logger.info('Training classifier: %d docs, %d unique labels', len(texts), len(set(labels)))
    vectorizer = TfidfVectorizer(min_df=config.min_df, max_df=config.max_df, ngram_range=config.ngram_range, stop_words=list(stop_words), token_pattern=SKLEARN_TOKEN_PATTERN)
    model = LogisticRegression(max_iter=config.max_iter, class_weight='balanced')
    pipeline = Pipeline([('vectorizer', vectorizer), ('model', model)])
    pipeline.fit(list(texts), list(labels))
    logger.info('Classifier training complete')
    return pipeline

def predict_labels(classifier: Pipeline, texts: Iterable[str]) -> list[str]:
    texts_list = list(texts)
    predictions = classifier.predict(texts_list).tolist()
    logger.debug('Predicted labels for %d documents', len(texts_list))
    return predictions

def save_classifier(path: Path, classifier: Pipeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, path)
    logger.info('Saved classifier to %s', path)

def load_classifier(path: Path) -> Pipeline:
    logger.info('Loading classifier from %s', path)
    return joblib.load(path)
