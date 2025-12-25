import re
try:
    import simplemma
except ImportError as exc:
    raise RuntimeError('Missing dependency: simplemma. Install requirements.txt and retry.') from exc
_TOKEN_RE = re.compile('[A-Za-zА-Яа-яЁё]+')

def lemmatize_text(text: str) -> str:
    lemmas = simplemma.text_lemmatizer(text.lower(), lang='ru')
    if not lemmas:
        return ''
    filtered = [lemma for lemma in lemmas if _TOKEN_RE.fullmatch(lemma)]
    return ' '.join(filtered)
