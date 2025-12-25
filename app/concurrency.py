import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable, TypeVar
logger = logging.getLogger(__name__)
T = TypeVar('T')

class ConcurrencyLimiter:

    def __init__(self, max_concurrent: int=3, timeout: float=300.0) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout = timeout
        self._max_concurrent = max_concurrent

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout)
            yield
        finally:
            self._semaphore.release()

    async def run_sync(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        async with self.acquire():
            logger.debug('Running %s with concurrency limiter (max=%d)', func.__name__, self._max_concurrent)
            return await asyncio.to_thread(func, *args, **kwargs)
_limiter: ConcurrencyLimiter | None = None

def get_limiter(max_concurrent: int=3, timeout: float=300.0) -> ConcurrencyLimiter:
    global _limiter
    if _limiter is None:
        _limiter = ConcurrencyLimiter(max_concurrent, timeout)
    return _limiter

def reset_limiter() -> None:
    global _limiter
    _limiter = None
