import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any
from telethon import TelegramClient, events, Button
from telethon.errors import FloodWaitError
from telethon.events import NewMessage, CallbackQuery
from app.config import PATHS
from app.exceptions import IngestError, PipelineError, TopicPredictorError
from app.ingest import ingest_channel
from app.pipeline import run_from_path
from app.visualization import generate_wordcloud, analyze_trends, format_trends
from app.doom_game import DoomGame

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)
_URL_RE = re.compile('(https?://t\\.me/[\\w_]+)/?')
_AT_RE = re.compile('@[\\w_]+')
_SEMAPHORE = asyncio.Semaphore(3)
_user_context: dict[int, dict[str, Any]] = {}
_doom_game = DoomGame()
_doom_tasks: dict[int, asyncio.Task] = {}
_DOOM_TICK_INTERVAL = 3.0

def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        raw = line.strip()
        if not raw or raw.startswith('#') or '=' not in raw:
            continue
        key, value = raw.split('=', 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if key not in os.environ or not os.environ.get(key):
            os.environ[key] = value

def _extract_channel(text: str) -> str | None:
    if not text:
        return None
    match = _URL_RE.search(text)
    if match:
        return match.group(1)
    match = _AT_RE.search(text)
    if match:
        return match.group(0)
    return None

def _get_main_menu() -> list[list[Button]]:
    return [[Button.inline('Популярные каналы', b'popular'), Button.inline('Помощь', b'help')]]

def _get_main_menu_with_last(channel: str) -> list[list[Button]]:
    channel_bytes = channel.encode('utf-8')
    channel_name = channel.split('/')[-1].replace('@', '')
    return [
        [Button.inline(f'🎯 {channel_name}', b'result:' + channel_bytes)],
        [Button.inline('Популярные каналы', b'popular'), Button.inline('Помощь', b'help')]
    ]

def _get_popular_channels() -> list[list[Button]]:
    return [
        [Button.inline('РИА Новости', b'channel:https://t.me/rian_ru')],
        [Button.inline('Mash', b'channel:https://t.me/breakingmash')],
        [Button.inline('RT на русском', b'channel:https://t.me/rt_russian')],
        [Button.inline('ТАСС', b'channel:https://t.me/tass_agency'), Button.inline('РБК', b'channel:https://t.me/rbc_news')],
        [Button.inline('Shot', b'channel:https://t.me/shot_shot'), Button.inline('Readovka', b'channel:https://t.me/readovkanews')],
        [Button.inline('Назад', b'back')]
    ]

def _get_channel_actions(channel: str) -> list[list[Button]]:
    channel_bytes = channel.encode('utf-8')
    return [[Button.inline('Предсказать', b'predict:' + channel_bytes), Button.inline('Обновить', b'refresh:' + channel_bytes)], [Button.inline('Назад', b'back')]]

def _get_result_actions(channel: str) -> list[list[Button]]:
    channel_bytes = channel.encode('utf-8')
    return [
        [Button.inline('☁️ Облако тем', b'cloud:' + channel_bytes), Button.inline('📈 Тренды', b'trends:' + channel_bytes)],
        [Button.inline('🔄 Обновить', b'refresh:' + channel_bytes), Button.inline('🏠 Меню', b'back')]
    ]

def _get_trends_actions(channel: str) -> list[list[Button]]:
    channel_bytes = channel.encode('utf-8')
    return [
        [Button.inline('🎯 К предсказанию', b'result:' + channel_bytes), Button.inline('☁️ Облако тем', b'cloud:' + channel_bytes)],
        [Button.inline('🏠 Меню', b'back')]
    ]

def _build_inline_keyboard(rows: list[list[tuple[str, str]]]) -> list[list[Button]]:
    return [[Button.inline(text, data.encode('utf-8')) for text, data in row] for row in rows]

def _stop_doom_task(user_id: int) -> None:
    task = _doom_tasks.pop(user_id, None)
    if task and not task.done():
        task.cancel()

def _format_result(result: dict[str, Any], channel: str = '') -> str:
    parts: list[str] = []
    if channel:
        channel_name = channel.split('/')[-1].replace('@', '')
        parts.append(f'📺 Канал: {channel_name}')
        parts.append(f'🔗 {channel}\n')
    warning = result.get('warning')
    if warning:
        parts.append(f'⚠️ {warning}')
    labeling = result.get('labeling')
    if labeling:
        label_names = {'seed': 'Базовая разметка', 'bootstrap_1': 'Расширенная разметка', 'bootstrap_2': 'Уточненная разметка', 'bootstrap_3': 'Финальная разметка'}
        for key, stats in labeling.items():
            coverage = stats.get('coverage', 0.0)
            name = label_names.get(key.lower(), key.capitalize())
            parts.append(f'📊 {name}: {coverage:.0%}')
    predictions = result.get('predictions', [])
    if not predictions:
        parts.append('\n❌ Нет предсказаний')
        return '\n'.join(parts)
    parts.append('\n🎯 Наиболее вероятные темы:')
    for idx, item in enumerate(predictions, start=1):
        label = item.get('label') or item.get('terms') or f"Тема {item.get('topic_id')}"
        prob = item.get('prob', 0.0)
        percent = prob * 100
        parts.append(f'{idx}. {label} — {percent:.0f}%')
    return '\n'.join(parts)

async def _handle(channel: str, refresh: bool) -> str:
    logger.info('Processing channel: %s (refresh=%s)', channel, refresh)
    async with _SEMAPHORE:
        path = await ingest_channel(channel, PATHS.data_dir, refresh=refresh)
        logger.info('Running pipeline on: %s', path)
        result = await asyncio.to_thread(run_from_path, path)
        logger.info('Pipeline finished, predictions: %d', len(result.get('predictions', [])))
    return _format_result(result, channel)

async def _safe_handle_with_buttons(event, channel: str, refresh: bool, *, edit: bool = True) -> None:
    user_id = event.sender_id
    msg = await (event.edit if edit else event.respond)('⏳ Загрузка данных...')
    try:
        async def update_status(text: str):
            try:
                await msg.edit(text)
            except:
                pass

        await update_status('⏳ Получение сообщений канала...')
        async with _SEMAPHORE:
            path = await ingest_channel(channel, PATHS.data_dir, refresh=refresh)
            await update_status('⏳ Анализ и обучение модели...')
            result = await asyncio.to_thread(run_from_path, path)

        _user_context[user_id] = {'channel': channel, 'result': result}
        reply = _format_result(result, channel)
        await msg.edit(reply, buttons=_get_result_actions(channel))
    except asyncio.TimeoutError:
        logger.error('Timeout processing channel: %s', channel)
        await msg.edit('❌ Превышено время ожидания. Канал может быть слишком большим.', buttons=_get_main_menu())
    except IngestError as exc:
        logger.warning('Ingest error for %s: %s', channel, exc)
        await msg.edit(f'❌ Не удается получить доступ к каналу: {exc}', buttons=_get_main_menu())
    except PipelineError as exc:
        logger.error('Pipeline error for %s: %s', channel, exc)
        await msg.edit(f'❌ Ошибка предсказания: {exc}', buttons=_get_main_menu())
    except TopicPredictorError as exc:
        logger.error('Application error for %s: %s', channel, exc)
        await msg.edit(f'❌ Ошибка: {exc}', buttons=_get_main_menu())
    except ValueError as exc:
        if 'After pruning, no terms remain' in str(exc):
            logger.warning('Channel incompatible: %s', channel)
            await msg.edit('❌ Этот канал несовместим с текущими настройками.\n\nПопробуйте другой канал из списка популярных.', buttons=_get_main_menu())
        else:
            logger.exception('Unexpected ValueError processing %s', channel)
            await msg.edit('❌ Произошла непредвиденная ошибка', buttons=_get_main_menu())
    except Exception as exc:
        logger.exception('Unexpected error processing %s', channel)
        await msg.edit('❌ Произошла непредвиденная ошибка', buttons=_get_main_menu())

async def main() -> None:
    _load_env(PATHS.base_dir / '.env')
    try:
        api_id = int(os.environ['TG_API_ID'])
        api_hash = os.environ['TG_API_HASH']
        bot_token = os.environ['TG_BOT_TOKEN']
    except KeyError as e:
        logger.error('Missing required environment variable: %s', e)
        raise SystemExit(1) from e
    session_path = str(PATHS.data_dir / 'bot_session')
    client = TelegramClient(session_path, api_id, api_hash)
    try:
        await client.start(bot_token=bot_token)
        me = await client.get_me()
        logger.info('Bot started: @%s', me.username)

        @client.on(events.NewMessage(pattern='^/start'))
        async def start_handler(event: NewMessage.Event) -> None:
            user_id = event.sender_id
            _user_context[user_id] = {}
            await event.respond('👋 Привет! Я бот для предсказания тем новостей.\n\nОтправьте ссылку на канал или выберите из популярных:', buttons=_get_main_menu())

        @client.on(events.NewMessage(pattern='^/doom'))
        async def doom_handler(event: NewMessage.Event) -> None:
            user_id = event.sender_id
            text = _doom_game.start(user_id)
            try:
                msg = await event.respond(text, buttons=_build_inline_keyboard(_doom_game.buttons(user_id)), parse_mode='html')
            except FloodWaitError as exc:
                logger.warning('DOOM start flood wait for %s: %s', user_id, exc.seconds)
                return
            except Exception:
                logger.exception('Failed to start DOOM for %s', user_id)
                msg = await event.respond(text, buttons=_build_inline_keyboard(_doom_game.buttons(user_id)))
            _stop_doom_task(user_id)

            async def _auto_loop(chat_id: int, message_id: int) -> None:
                try:
                    while _doom_game.is_active(user_id):
                        await asyncio.sleep(_DOOM_TICK_INTERVAL)
                        text_update = _doom_game.handle_action(user_id, 'tick')
                        buttons = _doom_game.buttons(user_id)
                        try:
                            await client.edit_message(chat_id, message_id, text_update, buttons=_build_inline_keyboard(buttons), parse_mode='html')
                        except FloodWaitError as exc:
                            await asyncio.sleep(exc.seconds + 0.5)
                        except Exception:
                            pass
                finally:
                    _stop_doom_task(user_id)

            _doom_tasks[user_id] = asyncio.create_task(_auto_loop(msg.chat_id, msg.id))

        @client.on(events.CallbackQuery(data=b'popular'))
        async def popular_handler(event: CallbackQuery.Event) -> None:
            await event.edit('📺 Выберите канал:', buttons=_get_popular_channels())

        @client.on(events.CallbackQuery(data=b'help'))
        async def help_handler(event: CallbackQuery.Event) -> None:
            help_text = '📖 Инструкция:\n\n1. Выберите канал из популярных или отправьте ссылку\n2. Нажмите "Предсказать" для анализа\n3. Используйте "Обновить" для свежих данных\n\n🔗 Формат ссылки:\nhttps://t.me/channel_name\nили @channel_name'
            await event.edit(help_text, buttons=_get_main_menu())

        @client.on(events.CallbackQuery(data=b'back'))
        async def back_handler(event: CallbackQuery.Event) -> None:
            user_id = event.sender_id
            ctx = _user_context.get(user_id, {})
            channel = ctx.get('channel')
            result = ctx.get('result')
            if channel and result:
                await event.edit('Выберите действие:', buttons=_get_main_menu_with_last(channel))
            else:
                await event.edit('Выберите действие:', buttons=_get_main_menu())

        @client.on(events.CallbackQuery(pattern=b'^channel:'))
        async def channel_select_handler(event: CallbackQuery.Event) -> None:
            channel = event.data.decode('utf-8').replace('channel:', '')
            user_id = event.sender_id
            _user_context[user_id] = {'channel': channel}
            await _safe_handle_with_buttons(event, channel, refresh=False, edit=True)

        @client.on(events.CallbackQuery(pattern=b'^predict:'))
        async def predict_callback_handler(event: CallbackQuery.Event) -> None:
            channel_bytes = event.data.replace(b'predict:', b'')
            channel = channel_bytes.decode('utf-8')
            await _safe_handle_with_buttons(event, channel, refresh=False, edit=True)

        @client.on(events.CallbackQuery(pattern=b'^refresh:'))
        async def refresh_callback_handler(event: CallbackQuery.Event) -> None:
            channel_bytes = event.data.replace(b'refresh:', b'')
            channel = channel_bytes.decode('utf-8')
            await _safe_handle_with_buttons(event, channel, refresh=True, edit=True)

        @client.on(events.CallbackQuery(pattern=b'^cloud:'))
        async def cloud_callback_handler(event: CallbackQuery.Event) -> None:
            user_id = event.sender_id
            channel_bytes = event.data.replace(b'cloud:', b'')
            channel = channel_bytes.decode('utf-8')
            ctx = _user_context.get(user_id, {})
            result = ctx.get('result')
            if not result:
                await event.answer('Сначала выполните предсказание', alert=True)
                return
            await event.answer('Генерирую облако тем...')
            topic_terms = result.get('topic_terms', [])
            if not topic_terms:
                await event.answer('Нет данных для облака тем', alert=True)
                return
            try:
                image_bytes = await asyncio.to_thread(generate_wordcloud, topic_terms)
                if image_bytes:
                    channel_name = channel.split('/')[-1].replace('@', '')
                    filename = f'{channel_name}_wordcloud.gif'
                    tmp_path = PATHS.data_dir / filename
                    tmp_path.write_bytes(image_bytes)
                    await event.respond(file=str(tmp_path))
                    tmp_path.unlink(missing_ok=True)
                else:
                    await event.answer('Не удалось сгенерировать облако', alert=True)
            except Exception as exc:
                logger.exception('Error generating wordcloud: %s', exc)
                await event.answer('Ошибка генерации облака', alert=True)

        @client.on(events.CallbackQuery(pattern=b'^trends:'))
        async def trends_callback_handler(event: CallbackQuery.Event) -> None:
            user_id = event.sender_id
            channel_bytes = event.data.replace(b'trends:', b'')
            channel = channel_bytes.decode('utf-8')
            ctx = _user_context.get(user_id, {})
            result = ctx.get('result')
            if not result:
                await event.answer('Сначала выполните предсказание', alert=True)
                return
            topics_seq = result.get('topics_seq', [])
            topic_terms = result.get('topic_terms', [])
            if not topics_seq or not topic_terms:
                await event.answer('Недостаточно данных для трендов', alert=True)
                return
            trends = analyze_trends(topics_seq, topic_terms)
            trends_text = format_trends(trends)
            try:
                await event.edit(trends_text, buttons=_get_trends_actions(channel))
            except Exception:
                await event.answer('Тренды уже отображены', alert=False)

        @client.on(events.CallbackQuery(pattern=b'^result:'))
        async def result_callback_handler(event: CallbackQuery.Event) -> None:
            user_id = event.sender_id
            channel_bytes = event.data.replace(b'result:', b'')
            channel = channel_bytes.decode('utf-8')
            ctx = _user_context.get(user_id, {})
            result = ctx.get('result')
            if not result:
                await event.answer('Сначала выполните предсказание', alert=True)
                return
            reply = _format_result(result, channel)
            try:
                await event.edit(reply, buttons=_get_result_actions(channel))
            except Exception:
                await event.answer('Предсказание уже отображено', alert=False)

        @client.on(events.CallbackQuery(pattern=b'^doom:'))
        async def doom_callback_handler(event: CallbackQuery.Event) -> None:
            user_id = event.sender_id
            action = event.data.decode('utf-8').replace('doom:', '')
            text = _doom_game.handle_action(user_id, action)
            buttons = _doom_game.buttons(user_id)
            try:
                await event.edit(text, buttons=_build_inline_keyboard(buttons), parse_mode='html')
            except FloodWaitError as exc:
                logger.warning('DOOM flood wait for %s: %s', user_id, exc.seconds)
                await event.answer(f'Подожди {exc.seconds} сек', alert=False)
            except Exception:
                logger.exception('Failed to update DOOM message for %s', user_id)
                try:
                    await event.edit(text, buttons=_build_inline_keyboard(buttons))
                except Exception:
                    await event.answer('Команда обработана', alert=False)
            if not _doom_game.is_active(user_id):
                _stop_doom_task(user_id)

        @client.on(events.NewMessage)
        async def message_handler(event: NewMessage.Event) -> None:
            if event.message.message.startswith('/'):
                return
            channel = _extract_channel(event.message.message)
            if not channel:
                await event.respond('❌ Не могу распознать канал.\n\nОтправьте ссылку вида:\nhttps://t.me/rbc_news\nили @rbc_news', buttons=_get_main_menu())
                return
            user_id = event.sender_id
            _user_context[user_id] = {'channel': channel}
            await _safe_handle_with_buttons(event, channel, refresh=False, edit=False)
        await client.run_until_disconnected()
    finally:
        await client.disconnect()
        logger.info('Bot stopped')
if __name__ == '__main__':
    asyncio.run(main())
