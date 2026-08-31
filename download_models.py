"""Однократно скачать модели, после чего анализ работает полностью офлайн."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
WHISPER_DIR = MODELS_DIR / "faster-whisper-small"
SEGMENTATION_DIR = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0"
EMBEDDING_PATH = MODELS_DIR / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

WHISPER_BASE_URL = "https://huggingface.co/Systran/faster-whisper-small/resolve/main"
WHISPER_FILES = (
    "config.json",
    "model.bin",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
    "vocabulary.txt",
)
EXPECTED_WHISPER_MODEL_SIZE = 483_546_902
EXPECTED_SEGMENTATION_SIZE = 1_540_506
EXPECTED_EMBEDDING_SIZE = 39_593_761

SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
    "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Скачивание {destination.name}...")
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urllib.request.urlopen(url, timeout=300) as response, temporary.open("wb") as target:
            expected_size = response.headers.get("Content-Length")
            shutil.copyfileobj(response, target, length=1024 * 1024)
        actual_size = temporary.stat().st_size
        if expected_size and actual_size != int(expected_size):
            raise RuntimeError(
                f"Файл {destination.name} скачан не полностью: {actual_size} из {expected_size} байт."
            )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    whisper_model = WHISPER_DIR / "model.bin"
    if not whisper_model.is_file() or whisper_model.stat().st_size != EXPECTED_WHISPER_MODEL_SIZE:
        print("Скачивание faster-whisper small (около 500 МБ)...")
        for filename in WHISPER_FILES:
            _download(f"{WHISPER_BASE_URL}/{filename}?download=true", WHISPER_DIR / filename)
    else:
        print("Whisper-модель уже установлена.")

    segmentation_model = SEGMENTATION_DIR / "model.int8.onnx"
    if not segmentation_model.is_file() or segmentation_model.stat().st_size != EXPECTED_SEGMENTATION_SIZE:
        with tempfile.TemporaryDirectory() as temp_dir:
            archive = Path(temp_dir) / "segmentation.tar.bz2"
            _download(SEGMENTATION_URL, archive)
            print("Распаковка модели сегментации...")
            with tarfile.open(archive, "r:bz2") as source:
                source.extractall(MODELS_DIR, filter="data")
    else:
        print("Модель сегментации уже установлена.")

    if not EMBEDDING_PATH.is_file() or EMBEDDING_PATH.stat().st_size != EXPECTED_EMBEDDING_SIZE:
        _download(EMBEDDING_URL, EMBEDDING_PATH)
    else:
        print("Модель голосовых эмбеддингов уже установлена.")

    print(f"Готово. Модели находятся в: {MODELS_DIR}")


if __name__ == "__main__":
    main()
