"""User-level config persisted at ~/.pixelpaws/user_config.json

Stores preferences that apply across all projects, such as
the global classifiers library folder path.
"""
import json
import os
from pathlib import Path

_DEFAULT_DIR = Path.home() / '.pixelpaws'
_CONFIG_FILE = _DEFAULT_DIR / 'user_config.json'
_DEFAULT_GLOBAL_CLF = _DEFAULT_DIR / 'global_classifiers'


def get_global_classifiers_folder() -> str:
    cfg = _load()
    return cfg.get('global_classifiers_folder', str(_DEFAULT_GLOBAL_CLF))


def set_global_classifiers_folder(path: str):
    cfg = _load()
    cfg['global_classifiers_folder'] = path
    _save(cfg)


def _load() -> dict:
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(cfg: dict):
    _DEFAULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=2)
