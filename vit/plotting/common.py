from __future__ import annotations

from datetime import datetime
from pathlib import Path


def timestamp_string() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def append_timestamp(path: Path, stamp: str | None = None) -> Path:
    stamp = timestamp_string() if stamp is None else stamp
    suffix = path.suffix or '.png'
    stem = path.stem if path.suffix else path.name
    return path.with_name(f'{stem}_{stamp}{suffix}')
