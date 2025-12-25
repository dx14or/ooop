from functools import lru_cache
from typing import Annotated
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class LabelingSettings(BaseModel):
    min_hits: Annotated[int, Field(ge=1, description='Minimum keyword hits for label')] = 1
    allow_ties: Annotated[bool, Field(description='Allow tied labels')] = True
    bootstrap_terms: Annotated[int, Field(ge=0, description='Number of bootstrap terms')] = 9
    bootstrap_min_df: Annotated[int, Field(ge=1, description='Min document frequency')] = 3
    bootstrap_max_df: Annotated[float, Field(gt=0, le=1, description='Max document frequency')] = 0.9
    bootstrap_rounds: Annotated[int, Field(ge=0, description='Number of bootstrap rounds')] = 1

class ClassifierSettings(BaseModel):
    min_docs: Annotated[int, Field(ge=10, description='Minimum documents for training')] = 300
    min_labels: Annotated[int, Field(ge=2, description='Minimum unique labels required')] = 6
    min_df: Annotated[int, Field(ge=1, description='Minimum document frequency')] = 4
    max_df: Annotated[float, Field(gt=0, le=1, description='Maximum document frequency')] = 0.9
    ngram_range: Annotated[tuple[int, int], Field(description='N-gram range')] = (1, 2)
    max_iter: Annotated[int, Field(ge=100, description='Maximum iterations')] = 500

    @model_validator(mode='after')
    def validate_ngram_range(self) -> 'ClassifierSettings':
        if self.ngram_range[0] > self.ngram_range[1]:
            raise ValueError('ngram_range[0] must be <= ngram_range[1]')
        return self

class TopicModelSettings(BaseModel):
    count: Annotated[int | None, Field(ge=2, description='Fixed topic count or None for auto')] = None
    min: Annotated[int, Field(ge=2, description='Minimum topics for auto-selection')] = 10
    max: Annotated[int, Field(ge=2, description='Maximum topics for auto-selection')] = 40
    step: Annotated[int, Field(ge=1, description='Step size for topic search')] = 5
    sample_size: Annotated[int, Field(ge=100, description='Sample size for selection')] = 5000
    test_size: Annotated[float, Field(gt=0, lt=1, description='Test split ratio')] = 0.1
    max_iter: Annotated[int, Field(ge=1, description='LDA max iterations')] = 5
    improvement_threshold: Annotated[float, Field(ge=0, description='Early stopping threshold')] = 0.01
    use_coherence: Annotated[bool, Field(description='Use coherence in selection')] = True
    coherence_top_terms: Annotated[int, Field(ge=1, description='Top terms for coherence')] = 8
    coherence_weight: Annotated[float, Field(ge=0, le=1, description='Coherence weight')] = 0.5

    @model_validator(mode='after')
    def validate_topic_range(self) -> 'TopicModelSettings':
        if self.min > self.max:
            raise ValueError('topic.min must be <= topic.max')
        return self

class PreprocessSettings(BaseModel):
    proper_noun_min_count: Annotated[int, Field(ge=1, description='Min count for proper noun')] = 20
    proper_noun_ratio: Annotated[float, Field(ge=0, le=1, description='Capitalization ratio')] = 0.85
    auto_stopwords_top_n: Annotated[int, Field(ge=0, description='Top N frequent terms')] = 30

class PredictionSettings(BaseModel):
    context_size: Annotated[int, Field(ge=1, description='Context window size')] = 5
    top_k: Annotated[int, Field(ge=1, description='Number of predictions')] = 3
    min_count: Annotated[int, Field(ge=1, description='Minimum context count')] = 2
    backoff_weights: Annotated[tuple[float, ...], Field(description='Backoff weights for n-gram sizes')] = (0.45, 0.25, 0.15, 0.1, 0.05)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='NTP_', env_nested_delimiter='__', env_file='.env', env_file_encoding='utf-8', extra='ignore')
    use_classifier: Annotated[bool, Field(description='Use classifier mode vs LDA')] = True
    model_version: Annotated[int, Field(ge=1, description='Model version for cache')] = 15
    labeling: LabelingSettings = Field(default_factory=LabelingSettings)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    topic: TopicModelSettings = Field(default_factory=TopicModelSettings)
    preprocess: PreprocessSettings = Field(default_factory=PreprocessSettings)
    prediction: PredictionSettings = Field(default_factory=PredictionSettings)

    @property
    def label_min_hits(self) -> int:
        return self.labeling.min_hits

    @property
    def label_allow_ties(self) -> bool:
        return self.labeling.allow_ties

    @property
    def label_bootstrap_terms(self) -> int:
        return self.labeling.bootstrap_terms

    @property
    def label_bootstrap_min_df(self) -> int:
        return self.labeling.bootstrap_min_df

    @property
    def label_bootstrap_max_df(self) -> float:
        return self.labeling.bootstrap_max_df

    @property
    def label_bootstrap_rounds(self) -> int:
        return self.labeling.bootstrap_rounds

    @property
    def classifier_min_docs(self) -> int:
        return self.classifier.min_docs

    @property
    def classifier_min_labels(self) -> int:
        return self.classifier.min_labels

    @property
    def classifier_min_df(self) -> int:
        return self.classifier.min_df

    @property
    def classifier_max_df(self) -> float:
        return self.classifier.max_df

    @property
    def classifier_ngram_range(self) -> tuple[int, int]:
        return self.classifier.ngram_range

    @property
    def classifier_max_iter(self) -> int:
        return self.classifier.max_iter

    @property
    def topic_count(self) -> int | None:
        return self.topic.count

    @property
    def topic_min(self) -> int:
        return self.topic.min

    @property
    def topic_max(self) -> int:
        return self.topic.max

    @property
    def topic_step(self) -> int:
        return self.topic.step

    @property
    def topic_sample_size(self) -> int:
        return self.topic.sample_size

    @property
    def topic_test_size(self) -> float:
        return self.topic.test_size

    @property
    def topic_max_iter(self) -> int:
        return self.topic.max_iter

    @property
    def topic_improvement_threshold(self) -> float:
        return self.topic.improvement_threshold

    @property
    def topic_use_coherence(self) -> bool:
        return self.topic.use_coherence

    @property
    def topic_coherence_top_terms(self) -> int:
        return self.topic.coherence_top_terms

    @property
    def topic_coherence_weight(self) -> float:
        return self.topic.coherence_weight

    @property
    def proper_noun_min_count(self) -> int:
        return self.preprocess.proper_noun_min_count

    @property
    def proper_noun_ratio(self) -> float:
        return self.preprocess.proper_noun_ratio

    @property
    def auto_stopwords_top_n(self) -> int:
        return self.preprocess.auto_stopwords_top_n

    @property
    def context_size(self) -> int:
        return self.prediction.context_size

    @property
    def prediction_top_k(self) -> int:
        return self.prediction.top_k

    @property
    def prediction_min_count(self) -> int:
        return self.prediction.min_count

    @property
    def prediction_backoff_weights(self) -> tuple[float, ...]:
        return self.prediction.backoff_weights

@lru_cache
def get_settings() -> Settings:
    return Settings()
