from functools import lru_cache
from pathlib import Path
from typing import Annotated
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

def _default_base_dir() -> Path:
    return Path(__file__).resolve().parents[2]

class Paths(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='NTP_', extra='ignore')
    base_dir: Annotated[Path, Field(description='Base application directory')] = Field(default_factory=_default_base_dir)

    @property
    def data_dir(self) -> Path:
        return self.base_dir / 'data'

    @property
    def models_dir(self) -> Path:
        return self.base_dir / 'models'

    @property
    def topics_path(self) -> Path:
        return self.base_dir / 'app' / 'topics.json'

    @property
    def web_dir(self) -> Path:
        return self.base_dir / 'web'

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

@lru_cache
def get_paths() -> Paths:
    return Paths()
