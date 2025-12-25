from .paths import Paths, get_paths
from .settings import Settings, get_settings
PATHS = get_paths()
SETTINGS = get_settings()
__all__ = ['Paths', 'Settings', 'get_paths', 'get_settings', 'PATHS', 'SETTINGS']
