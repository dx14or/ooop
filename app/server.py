import asyncio
import logging
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .config import PATHS
from .exceptions import IngestError, PipelineError, SecurityError
from .ingest import ingest_channel
from .pipeline import run_from_path
from .security import validate_data_path, sanitize_channel_url
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    channel_url: str | None = None
    data_path: str | None = None
app = FastAPI(title='News Topic Predictor', description='Predict next topic for Telegram news channels', version='1.0.0')
if (PATHS.base_dir / 'web').exists():
    app.mount('/static', StaticFiles(directory=PATHS.base_dir / 'web'), name='static')

@app.get('/')
def root() -> FileResponse:
    index_path = PATHS.base_dir / 'web' / 'index.html'
    if not index_path.exists():
        raise HTTPException(status_code=404, detail='Web interface not found')
    return FileResponse(index_path)

@app.get('/health')
def health() -> dict[str, bool]:
    return {'ok': True}

@app.post('/predict')
async def predict(request: PredictRequest) -> dict[str, Any]:
    if request.data_path:
        try:
            path = validate_data_path(request.data_path, PATHS.data_dir)
        except SecurityError as exc:
            logger.warning('Security error: %s', exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not path.exists():
            raise HTTPException(status_code=404, detail='data_path not found')
        logger.info('Running prediction on local file: %s', path)
        try:
            result = await asyncio.to_thread(run_from_path, path)
        except PipelineError as exc:
            logger.error('Pipeline error: %s', exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result
    if request.channel_url:
        try:
            channel = sanitize_channel_url(request.channel_url)
        except SecurityError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info('Ingesting channel: %s', channel)
        try:
            path = await ingest_channel(channel, PATHS.data_dir)
        except IngestError as exc:
            logger.error('Ingest error: %s', exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        logger.info('Running prediction on ingested data: %s', path)
        try:
            result = await asyncio.to_thread(run_from_path, path)
        except PipelineError as exc:
            logger.error('Pipeline error: %s', exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return result
    raise HTTPException(status_code=400, detail='Provide data_path or channel_url')
