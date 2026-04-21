from __future__ import annotations

import argparse
from pathlib import Path

from .config import SortConfig
from .sorter import PhotoSorter, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="photo-sorter",
        description="Сортировка фото шкафов по шильдикам, Excel-реестру и папкам объектов",
    )
    parser.add_argument("--photos-dir", required=True, help="Папка с входными фото")
    parser.add_argument("--excel-file", required=True, help="Excel-реестр .xlsx")
    parser.add_argument("--objects-root", required=True, help="Корень каталога Объекты")
    parser.add_argument("--output-dir", required=True, help="Папка результата")

    parser.add_argument("--sheet-name", default="Итого")
    parser.add_argument("--serial-column", default="Заводской №")
    parser.add_argument("--order-column", default="№заказа")
    parser.add_argument("--sort-by", default="exif_then_file", choices=["exif_then_file", "file_time", "name"])

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--copy", action="store_true", help="Копировать файлы (по умолчанию)")
    mode.add_argument("--move", action="store_true", help="Перемещать файлы")

    parser.add_argument("--serial-subfolder", action="store_true", help="Создавать подпапки по заводскому номеру")
    parser.add_argument("--no-remember-processed", action="store_true", help="Не запоминать уже обработанные файлы")
    parser.add_argument("--skip-before-first-label", action="store_true", help="Пропускать фото до первого шильдика")
    parser.add_argument("--no-qr", action="store_true", help="Отключить сканирование QR-кода")
    parser.add_argument("--fast-mode", action="store_true", help="Быстрый режим: QR -> OCR полного кадра -> повороты без тяжелых зон")
    parser.add_argument(
        "--no-verify-by-registry",
        action="store_true",
        help="Не перепроверять кандидаты по Excel-реестру при выборе серийного номера",
    )

    return parser


def run_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = SortConfig(
        photos_dir=Path(args.photos_dir),
        excel_file=Path(args.excel_file),
        objects_root=Path(args.objects_root),
        output_dir=Path(args.output_dir),
        sheet_name=args.sheet_name,
        serial_column=args.serial_column,
        order_column=args.order_column,
        mode="move" if args.move else "copy",
        sort_by=args.sort_by,
        create_serial_subfolder=args.serial_subfolder,
        remember_processed=not args.no_remember_processed,
        put_before_first_label_to_unrecognized=not args.skip_before_first_label,
        include_qr_scan=not args.no_qr,
        fast_mode=args.fast_mode,
        verify_by_registry=not args.no_verify_by_registry,
    )
    validate_config(config)

    sorter = PhotoSorter(config, logger=print)
    sorter.run()


if __name__ == "__main__":
    run_cli()
