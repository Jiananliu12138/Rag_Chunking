"""
在导入现有流水线模块之前，将相关目录注入 sys.path。
调用 ensure_paths() 一次即可，幂等操作。
"""
import sys
from app.config import get_settings


def ensure_paths() -> None:
    settings = get_settings()
    for p in settings.python_paths:
        if p not in sys.path:
            sys.path.insert(0, p)
