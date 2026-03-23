from __future__ import annotations

import sys

from app.config import get_settings


def ensure_paths() -> None:
    settings = get_settings()
    for path in settings.python_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
