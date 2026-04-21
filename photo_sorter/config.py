from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


FileMode = Literal["copy", "move"]
SortBy = Literal["exif_then_file", "file_time", "name"]


@dataclass
class SortConfig:
    photos_dir: Path
    excel_file: Path
    objects_root: Path
    output_dir: Path
    sheet_name: str = "Итого"
    serial_column: str = "Заводской №"
    order_column: str = "№заказа"
    mode: FileMode = "copy"
    sort_by: SortBy = "exif_then_file"
    put_before_first_label_to_unrecognized: bool = True
    create_serial_subfolder: bool = False
    remember_processed: bool = True
    include_qr_scan: bool = True
    fast_mode: bool = False
    verify_by_registry: bool = True
    log_file_name: str = "process_log.csv"
    state_file_name: str = "processed_files.json"

    label_keywords: tuple[str, ...] = field(
        default_factory=lambda: (
            "завод",
            "завод. №",
            "завод №",
            "зав. №",
            "год выпуска",
            "степень защиты",
            "ту",
            "eac",
            "номинальное напряжение",
            "номинальный ток",
            "кол-во отход.лин",
        )
    )

    # Области для OCR: полное изображение, крупные части, углы.
    region_specs: tuple[tuple[float, float, float, float], ...] = field(
        default_factory=lambda: (
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 0.7),
            (0.0, 0.3, 1.0, 0.7),
            (0.0, 0.0, 0.7, 1.0),
            (0.3, 0.0, 0.7, 1.0),
            (0.0, 0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5, 0.5),
            (0.0, 0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5, 0.5),
            (0.2, 0.2, 0.6, 0.6),
        )
    )

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "_НЕ_РАСПОЗНАНО").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "_НЕТ_В_РЕЕСТРЕ").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "_НЕОДНОЗНАЧНО_ОБЪЕКТ").mkdir(parents=True, exist_ok=True)
