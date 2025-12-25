from typing import Any

class TopicPredictorError(Exception):

    def __init__(self, message: str, details: dict[str, Any] | None=None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

class ConfigurationError(TopicPredictorError):
    pass

class IngestError(TopicPredictorError):
    pass

class PipelineError(TopicPredictorError):
    pass

class ModelError(PipelineError):
    pass

class CacheError(TopicPredictorError):
    pass

class ValidationError(TopicPredictorError):
    pass

class SecurityError(TopicPredictorError):
    pass
