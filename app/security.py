import logging
from pathlib import Path
from typing import FrozenSet
from .exceptions import SecurityError
logger = logging.getLogger(__name__)
ALLOWED_DATA_EXTENSIONS: FrozenSet[str] = frozenset({'.jsonl', '.jsonl.gz', '.gz', '.json'})

def validate_data_path(user_path: str, allowed_base: Path, allowed_extensions: FrozenSet[str]=ALLOWED_DATA_EXTENSIONS) -> Path:
    if not user_path:
        raise SecurityError('Empty path provided')
    try:
        path = Path(user_path).resolve()
    except (ValueError, OSError) as e:
        logger.warning('Invalid path format: %s', user_path)
        raise SecurityError(f'Invalid path format: {user_path}') from e
    allowed_base_resolved = allowed_base.resolve()
    try:
        path.relative_to(allowed_base_resolved)
    except ValueError:
        logger.warning('Path traversal attempt: %s (base: %s)', user_path, allowed_base)
        raise SecurityError(f'Path must be within {allowed_base}, got: {user_path}')
    suffixes = ''.join(path.suffixes).lower()
    if not any((suffixes.endswith(ext) for ext in allowed_extensions)):
        logger.warning('Invalid extension: %s (allowed: %s)', suffixes, allowed_extensions)
        raise SecurityError(f"Invalid file extension '{suffixes}'. Allowed: {sorted(allowed_extensions)}")
    return path

def sanitize_channel_url(url: str) -> str:
    if not url:
        raise SecurityError('Empty channel URL')
    url = url.strip()
    if url.startswith('@'):
        username = url[1:]
        if not username.replace('_', '').isalnum():
            raise SecurityError(f'Invalid channel username: {url}')
        return url
    if 't.me/' in url:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url
    raise SecurityError(f'Invalid channel format: {url}. Use @username or https://t.me/channel')
