from dataclasses import dataclass, field
from typing import Callable
from .config import Paths, Settings, get_paths, get_settings

@dataclass
class ServiceContainer:
    settings_provider: Callable[[], Settings] = field(default=get_settings)
    paths_provider: Callable[[], Paths] = field(default=get_paths)

    @property
    def settings(self) -> Settings:
        return self.settings_provider()

    @property
    def paths(self) -> Paths:
        return self.paths_provider()
_container: ServiceContainer | None = None

def get_container() -> ServiceContainer:
    global _container
    if _container is None:
        _container = ServiceContainer()
    return _container

def set_container(container: ServiceContainer) -> None:
    global _container
    _container = container

def reset_container() -> None:
    global _container
    _container = None
