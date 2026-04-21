from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import SortConfig
from .excel_registry import PassportRegistry, RegistryError
from .image_io import list_supported_images, sort_images
from .object_finder import ObjectFolderFinder
from .ocr_engine import LabelRecognizer, SerialCandidate


@dataclass
class Group:
    files: list[Path]
    destination: Path
    status: str
    serial: str | None = None
    order_number: str | None = None
    comment: str = ""
    serial_reason: str = ""
    serial_candidates: str = ""
    full_frame_candidates: str = ""
    region_candidates: str = ""
    target_zone_candidates: str = ""
    globally_rejected_candidates: str = ""
    registry_matches: str = ""
    final_decision_reason: str = ""


class PhotoSorter:
    def __init__(self, config: SortConfig, logger: Callable[[str], None] | None = None) -> None:
        self.config = config
        self.logger = logger or (lambda msg: None)

        self.registry = PassportRegistry(
            excel_file=config.excel_file,
            sheet_name=config.sheet_name,
            serial_column=config.serial_column,
            order_column=config.order_column,
        )
        self.object_finder = ObjectFolderFinder(config.objects_root)
        self.recognizer = LabelRecognizer(config)

        self.process_log_path = config.output_dir / config.log_file_name
        self.state_path = config.output_dir / config.state_file_name
        self.processed_files: set[str] = set()

    def _load_state(self) -> None:
        if self.config.remember_processed and self.state_path.exists():
            try:
                self.processed_files = set(json.loads(self.state_path.read_text(encoding="utf-8")))
            except Exception:
                self.processed_files = set()

    def _save_state(self) -> None:
        if not self.config.remember_processed:
            return
        self.state_path.write_text(
            json.dumps(sorted(self.processed_files), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _format_candidates(candidates: list[SerialCandidate]) -> str:
        parts: list[str] = []
        for cand in candidates[:12]:
            reject = cand.rejected_reason or "ok"
            parts.append(f"{cand.serial}|p={cand.priority}|{cand.source}|{reject}")
        return "; ".join(parts)

    @staticmethod
    def _join(items: list[str]) -> str:
        return "; ".join(items[:20])

    def _append_log(self, rows: list[dict[str, str]]) -> None:
        write_header = not self.process_log_path.exists()
        fields = [
            "дата/время обработки",
            "имя файла",
            "определен ли шильдик",
            "распознанный заводской номер",
            "найден ли номер в реестре",
            "найденный №заказа",
            "найдено ли имя папки объекта",
            "итоговая папка назначения",
            "статус",
            "причина выбора serial",
            "кандидаты serial",
            "full_frame_candidates",
            "region_candidates",
            "target_zone_candidates",
            "globally_rejected_candidates",
            "registry_matches",
            "final_decision_reason",
            "комментарий / текст ошибки",
        ]
        with self.process_log_path.open("a", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _copy_or_move(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if self.config.mode == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)

    def _flush_group(self, group: Group, log_rows: list[dict[str, str]]) -> None:
        for file in group.files:
            target = group.destination / file.name
            try:
                self._copy_or_move(file, target)
                self.processed_files.add(str(file.resolve()))
                status = group.status
                comment = group.comment
            except Exception as exc:
                status = "FILE_ERROR"
                comment = str(exc)

            log_rows.append(
                {
                    "дата/время обработки": datetime.now().isoformat(sep=" ", timespec="seconds"),
                    "имя файла": file.name,
                    "определен ли шильдик": "Да" if (group.serial or group.final_decision_reason) else "Нет",
                    "распознанный заводской номер": group.serial or "",
                    "найден ли номер в реестре": "Да" if group.order_number else "Нет",
                    "найденный №заказа": group.order_number or "",
                    "найдено ли имя папки объекта": "Да" if group.destination.name not in {"_НЕ_РАСПОЗНАНО", "_НЕТ_В_РЕЕСТРЕ", "_НЕОДНОЗНАЧНО_ОБЪЕКТ"} else "Нет",
                    "итоговая папка назначения": str(group.destination),
                    "статус": status,
                    "причина выбора serial": group.serial_reason,
                    "кандидаты serial": group.serial_candidates,
                    "full_frame_candidates": group.full_frame_candidates,
                    "region_candidates": group.region_candidates,
                    "target_zone_candidates": group.target_zone_candidates,
                    "globally_rejected_candidates": group.globally_rejected_candidates,
                    "registry_matches": group.registry_matches,
                    "final_decision_reason": group.final_decision_reason,
                    "комментарий / текст ошибки": comment,
                }
            )

    def run(self) -> None:
        self.config.ensure_dirs()
        self._load_state()
        self.registry.load()
        registry_serials = set(self.registry.serial_to_order.keys())

        photos = sort_images(list_supported_images(self.config.photos_dir), strategy=self.config.sort_by)
        photos = [p for p in photos if str(p.resolve()) not in self.processed_files]

        self.logger(f"Найдено файлов для обработки: {len(photos)}")

        current_group: Group | None = None
        log_rows: list[dict[str, str]] = []

        for idx, image_path in enumerate(photos, start=1):
            self.logger(f"[{idx}/{len(photos)}] OCR: {image_path.name}")
            scan = self.recognizer.scan_file(image_path, registry_serials=registry_serials)
            candidate_view = self._format_candidates(scan.candidates)

            self.logger(f"    result_full_frame: {scan.result_full_frame}")
            self.logger(f"    result_region_scan: {scan.result_region_scan}")
            self.logger(f"    result_targeted_bottom_right: {scan.result_targeted_bottom_right}")
            self.logger(f"    chosen_serial: {scan.serial or '-'}")
            self.logger(f"    chosen_reason: {scan.chosen_reason or '-'}")

            if scan.is_label:
                if current_group:
                    self._flush_group(current_group, log_rows)

                if scan.serial:
                    order_number = self.registry.find_order_by_serial(scan.serial)
                    if not order_number:
                        destination = self.config.output_dir / "_НЕТ_В_РЕЕСТРЕ"
                        status = "SERIAL_NOT_IN_REGISTRY"
                        comment = f"Серийный номер {scan.serial} отсутствует в реестре"
                    else:
                        obj_result = self.object_finder.find_folder(order_number)
                        if obj_result.status == "OBJECT_FOLDER_AMBIGUOUS":
                            destination = self.config.output_dir / "_НЕОДНОЗНАЧНО_ОБЪЕКТ" / order_number
                            status = obj_result.status
                            comment = obj_result.comment
                        else:
                            destination = self.config.output_dir / obj_result.folder_name
                            status = "OK" if obj_result.status == "OK" else "OBJECT_FOLDER_NOT_FOUND"
                            comment = obj_result.comment

                        if self.config.create_serial_subfolder:
                            destination = destination / scan.serial
                else:
                    order_number = None
                    if scan.final_decision_reason == "no_registry_match":
                        destination = self.config.output_dir / "_НЕТ_В_РЕЕСТРЕ"
                        status = "SERIAL_NOT_IN_REGISTRY"
                        comment = "Не найден кандидат, присутствующий в Excel-реестре"
                    else:
                        destination = self.config.output_dir / "_НЕ_РАСПОЗНАНО"
                        status = "SERIAL_NOT_RECOGNIZED"
                        comment = "Шильдик обнаружен, но заводской номер не извлечен"

                current_group = Group(
                    files=[image_path],
                    destination=destination,
                    status=status,
                    serial=scan.serial,
                    order_number=order_number,
                    comment=comment,
                    serial_reason=scan.chosen_reason,
                    serial_candidates=candidate_view,
                    full_frame_candidates=self._join(scan.full_frame_candidates),
                    region_candidates=self._join(scan.region_candidates),
                    target_zone_candidates=self._join(scan.target_zone_candidates),
                    globally_rejected_candidates=self._join(scan.globally_rejected_candidates),
                    registry_matches=self._join(scan.registry_matches),
                    final_decision_reason=scan.final_decision_reason,
                )
                self.logger(f" -> Шильдик: serial={scan.serial}, status={status}")
                continue

            if current_group is None:
                if self.config.put_before_first_label_to_unrecognized:
                    current_group = Group(
                        files=[image_path],
                        destination=self.config.output_dir / "_НЕ_РАСПОЗНАНО",
                        status="SERIAL_NOT_RECOGNIZED",
                        comment="Фото до первого шильдика или шильдик не найден",
                    )
                else:
                    log_rows.append(
                        {
                            "дата/время обработки": datetime.now().isoformat(sep=" ", timespec="seconds"),
                            "имя файла": image_path.name,
                            "определен ли шильдик": "Нет",
                            "распознанный заводской номер": "",
                            "найден ли номер в реестре": "Нет",
                            "найденный №заказа": "",
                            "найдено ли имя папки объекта": "Нет",
                            "итоговая папка назначения": "",
                            "статус": "SKIPPED_BEFORE_FIRST_LABEL",
                            "причина выбора serial": "",
                            "кандидаты serial": "",
                            "full_frame_candidates": "",
                            "region_candidates": "",
                            "target_zone_candidates": "",
                            "globally_rejected_candidates": "",
                            "registry_matches": "",
                            "final_decision_reason": "",
                            "комментарий / текст ошибки": "Пропущено по настройке",
                        }
                    )
                continue

            current_group.files.append(image_path)

        if current_group:
            self._flush_group(current_group, log_rows)

        self._append_log(log_rows)
        self._save_state()
        self.logger("Готово.")


def validate_config(config: SortConfig) -> None:
    for p, name in (
        (config.photos_dir, "Папка с фото"),
        (config.objects_root, "Корневая папка объектов"),
    ):
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"{name} не найдена: {p}")
    if not config.excel_file.exists():
        raise FileNotFoundError(f"Excel-файл не найден: {config.excel_file}")

    try:
        PassportRegistry(config.excel_file, config.sheet_name, config.serial_column, config.order_column).load()
    except RegistryError:
        raise
