from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import ExifTags, Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

SUPPORTED_EXTENSIONS = {".heic", ".jpg", ".jpeg", ".png"}


_DATETIME_TAGS = ("DateTimeOriginal", "DateTimeDigitized", "DateTime")


def list_supported_images(path: Path) -> list[Path]:
    return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]


def _parse_exif_datetime(value: str) -> dt.datetime | None:
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(value.strip(), fmt)
        except Exception:
            continue
    return None


def extract_shot_datetime(path: Path) -> dt.datetime | None:
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if exif:
                for key in exif:
                    tag_name = ExifTags.TAGS.get(key, str(key))
                    if tag_name in _DATETIME_TAGS:
                        parsed = _parse_exif_datetime(str(exif.get(key)))
                        if parsed:
                            return parsed
    except Exception:
        return None
    return None


def filename_sort_hint(path: Path) -> tuple[int, str]:
    nums = re.findall(r"\d+", path.name)
    return (int(nums[0]) if nums else 10**12, path.name.lower())


def sort_images(paths: Iterable[Path], strategy: str = "exif_then_file") -> list[Path]:
    decorated = []
    for p in paths:
        exif_dt = extract_shot_datetime(p)
        stat = p.stat()
        file_dt = dt.datetime.fromtimestamp(stat.st_mtime)
        decorated.append((p, exif_dt, file_dt, filename_sort_hint(p)))

    if strategy == "name":
        key_fn = lambda row: (row[3], row[2])
    elif strategy == "file_time":
        key_fn = lambda row: (row[2], row[3])
    else:
        key_fn = lambda row: (row[1] is None, row[1] or row[2], row[2], row[3])

    return [x[0] for x in sorted(decorated, key=key_fn)]


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        normalized = ImageOps.exif_transpose(img)
        rgb = normalized.convert("RGB")
        return np.array(rgb)
