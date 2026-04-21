from __future__ import annotations

from pathlib import Path

import pandas as pd


class RegistryError(Exception):
    pass


class PassportRegistry:
    def __init__(self, excel_file: Path, sheet_name: str, serial_column: str, order_column: str) -> None:
        self.excel_file = excel_file
        self.sheet_name = sheet_name
        self.serial_column = serial_column
        self.order_column = order_column
        self.serial_to_order: dict[str, str] = {}

    @staticmethod
    def _normalize(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if text.endswith(".0") and text.replace(".", "", 1).isdigit():
            text = text[:-2]
        return text

    def load(self) -> None:
        try:
            df = pd.read_excel(self.excel_file, sheet_name=self.sheet_name, dtype=str, engine="openpyxl")
        except ValueError as exc:
            raise RegistryError(f"Лист '{self.sheet_name}' не найден в Excel.") from exc

        missing = [c for c in (self.serial_column, self.order_column) if c not in df.columns]
        if missing:
            cols = ", ".join(missing)
            raise RegistryError(f"В Excel не найдены колонки: {cols}.")

        mapping: dict[str, str] = {}
        for _, row in df.iterrows():
            serial = self._normalize(row.get(self.serial_column, ""))
            order = self._normalize(row.get(self.order_column, ""))
            if serial and order:
                mapping[serial] = order

        self.serial_to_order = mapping

    def find_order_by_serial(self, serial: str) -> str | None:
        return self.serial_to_order.get(self._normalize(serial))
