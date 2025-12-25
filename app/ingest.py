import asyncio
import gzip
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable
from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, PhoneCodeExpiredError, PhoneCodeInvalidError, SessionPasswordNeededError, UsernameInvalidError, UsernameNotOccupiedError
from .exceptions import IngestError
logger = logging.getLogger(__name__)
DEFAULT_MESSAGE_LIMIT = 10000

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == '.gz' else open
    rows: list[dict[str, Any]] = []
    try:
        with opener(path, 'rt', encoding='utf-8') as fp:
            for line_num, line in enumerate(fp, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning('Invalid JSON at line %d: %s', line_num, e)
    except (OSError, gzip.BadGzipFile) as e:
        raise IngestError(f'Failed to read {path}: {e}') from e
    return rows

def save_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    opener = gzip.open if path.suffix == '.gz' else open
    path.parent.mkdir(parents=True, exist_ok=True)
    with opener(path, 'wt', encoding='utf-8') as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + '\n')

def _env_required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise IngestError(f'Missing required env var: {name}')
    return value

def _channel_slug(channel_url: str) -> str:
    channel_url = channel_url.strip()
    if channel_url.startswith('@'):
        channel_url = channel_url[1:]
    if 't.me/' in channel_url:
        channel_url = channel_url.split('t.me/')[-1]
    channel_url = channel_url.split('?')[0].strip('/')
    slug = channel_url.split('/')[-1]
    return re.sub('[^A-Za-z0-9_-]+', '_', slug) or 'channel'

async def _ensure_login(client: TelegramClient, phone: Optional[str], code: Optional[str], password: Optional[str], code_hash_path: Path) -> None:
    if await client.is_user_authorized():
        return
    if not phone:
        raise IngestError('Not logged in. Set TG_PHONE to request a login code.')
    if code:
        if not code_hash_path.exists():
            raise IngestError('TG_CODE provided but no stored code hash. Run once without TG_CODE to request a new code.')
        phone_code_hash = code_hash_path.read_text().strip()
        try:
            await client.sign_in(phone=phone, code=code, phone_code_hash=phone_code_hash)
        except SessionPasswordNeededError:
            if not password:
                raise IngestError('Account has 2FA password. Set TG_PASSWORD and rerun.')
            await client.sign_in(password=password)
        except (PhoneCodeInvalidError, PhoneCodeExpiredError) as exc:
            raise IngestError(f'Login code failed ({exc}). Request a new code and rerun.')
        code_hash_path.unlink(missing_ok=True)
        return
    sent = await client.send_code_request(phone)
    code_hash_path.write_text(sent.phone_code_hash)
    raise IngestError('Login code sent. Check Telegram, set TG_CODE=<code>, and rerun soon.')

async def _export_channel(channel: str, output_path: Path, client: TelegramClient, limit: int=DEFAULT_MESSAGE_LIMIT) -> int:
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix('.tmp.gz')
    try:
        with gzip.open(tmp_path, 'wt', encoding='utf-8') as fp:
            try:
                async for msg in client.iter_messages(channel, limit=limit, reverse=True):
                    payload = {'id': msg.id, 'date': msg.date.isoformat() if msg.date else None, 'text': msg.message, 'views': msg.views, 'forwards': msg.forwards, 'replies': msg.replies.replies if msg.replies else None, 'media': bool(msg.media)}
                    fp.write(json.dumps(payload, ensure_ascii=False) + '\n')
                    count += 1
                    if count % 1000 == 0:
                        logger.info('Exported %d messages from %s', count, channel)
            except (ChannelPrivateError, UsernameInvalidError, UsernameNotOccupiedError) as e:
                raise IngestError(f"Cannot access channel '{channel}': {e}") from e
        tmp_path.replace(output_path)
        logger.info('Finished exporting %d messages to %s', count, output_path)
    except IngestError:
        tmp_path.unlink(missing_ok=True)
        raise
    except (OSError, IOError) as e:
        tmp_path.unlink(missing_ok=True)
        raise IngestError(f'Failed to write export file: {e}') from e
    return count

async def ingest_channel(channel_url: str, data_dir: Path, refresh: bool=False) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    slug = _channel_slug(channel_url)
    output_path = data_dir / f'{slug}.jsonl.gz'
    env_refresh = os.environ.get('TG_REFRESH') == '1'
    if output_path.exists() and (not (refresh or env_refresh)):
        logger.info('Using cached data: %s', output_path)
        return output_path
    logger.info('Ingesting channel: %s', channel_url)
    api_id = int(_env_required('TG_API_ID'))
    api_hash = _env_required('TG_API_HASH')
    phone = os.environ.get('TG_PHONE')
    code = os.environ.get('TG_CODE')
    password = os.environ.get('TG_PASSWORD')
    session_label = os.environ.get('TG_SESSION', 'tg_user')
    session_name = str(data_dir / session_label)
    code_hash_path = data_dir / f'.tg_code_hash_{session_label}'
    client = TelegramClient(session_name, api_id, api_hash)
    await client.connect()
    try:
        await _ensure_login(client, phone, code, password, code_hash_path)
        await _export_channel(channel_url, output_path, client)
    finally:
        await client.disconnect()
    return output_path

def ingest_channel_sync(channel_url: str, data_dir: Path, refresh: bool=False) -> Path:
    return asyncio.run(ingest_channel(channel_url, data_dir, refresh=refresh))
