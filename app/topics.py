import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from .config import PATHS

@dataclass(frozen=True)
class TopicDefinition:
    id: str
    title: str
    keywords: set[str]

def _default_topics_path(path: Path | None) -> Path:
    return path or PATHS.base_dir / 'app' / 'topics.json'

def load_topics(path: Path | None=None) -> list[TopicDefinition]:
    topics_path = _default_topics_path(path)
    data = json.loads(topics_path.read_text(encoding='utf-8'))
    topics: list[TopicDefinition] = []
    seen_ids: set[str] = set()
    for item in data:
        topic_id = str(item['id']).strip()
        title = str(item['title']).strip()
        keywords = {str(word).strip().lower() for word in item.get('keywords', []) if word}
        if not topic_id or topic_id in seen_ids:
            raise ValueError(f'Duplicate or empty topic id: {topic_id!r}')
        if not title:
            raise ValueError(f'Missing title for topic id: {topic_id!r}')
        if not keywords:
            raise ValueError(f'Missing keywords for topic id: {topic_id!r}')
        seen_ids.add(topic_id)
        topics.append(TopicDefinition(id=topic_id, title=title, keywords=keywords))
    return topics

def topics_signature(path: Path | None=None) -> str:
    topics_path = _default_topics_path(path)
    payload = topics_path.read_bytes()
    return hashlib.sha256(payload).hexdigest()[:12]

def topic_ids(topics: Iterable[TopicDefinition]) -> list[str]:
    return [topic.id for topic in topics]
