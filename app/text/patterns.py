import re
from typing import Final
TOKEN_PATTERN: Final[str] = '[A-Za-zА-Яа-яЁё]+'
TOKEN_RE: Final[re.Pattern[str]] = re.compile(TOKEN_PATTERN)
SKLEARN_TOKEN_PATTERN: Final[str] = '(?u)\\b[а-яёa-z][а-яёa-z]+\\b'
URL_PATTERN: Final[str] = 'https?://\\S+|www\\.\\S+'
URL_RE: Final[re.Pattern[str]] = re.compile(URL_PATTERN)
TELEGRAM_MENTION_RE: Final[re.Pattern[str]] = re.compile('@[\\w_]+')

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)

def tokenize_lower(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]

def remove_urls(text: str) -> str:
    return URL_RE.sub(' ', text)
