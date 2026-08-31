"""Минимальный пример запуска локального анализа разговора."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ANALYZER_FILE = PROJECT_DIR / "voice description.py"


def _load_analyzer_module():
    """Загрузить модуль, имя файла которого содержит пробел."""
    spec = importlib.util.spec_from_file_location("voice_description", ANALYZER_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль: {ANALYZER_FILE}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_example() -> Path:
    """Разобрать тестовую запись voice example.wav и вернуть путь к Excel."""
    analyzer = _load_analyzer_module()
    audio_file = PROJECT_DIR / "voice example.wav"
    result = analyzer.analyze_call(str(audio_file))
    print(f"Готово: {result}")
    return result


if __name__ == "__main__":
    run_example()
