from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ObjectMatchResult:
    status: str
    folder_name: str
    comment: str = ""


class ObjectFolderFinder:
    def __init__(self, objects_root: Path) -> None:
        self.objects_root = objects_root

    def find_folder(self, order_number: str) -> ObjectMatchResult:
        escaped = re.escape(order_number)
        pattern = re.compile(rf"^{escaped}(?:[\s\-]|$)")
        matches = [p for p in self.objects_root.iterdir() if p.is_dir() and pattern.match(p.name)]

        if len(matches) == 1:
            return ObjectMatchResult("OK", matches[0].name)
        if len(matches) == 0:
            return ObjectMatchResult("OBJECT_FOLDER_NOT_FOUND", order_number, "Папка объекта не найдена")

        return ObjectMatchResult(
            "OBJECT_FOLDER_AMBIGUOUS",
            order_number,
            f"Найдено несколько папок: {', '.join(m.name for m in matches[:5])}",
        )
