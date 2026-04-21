from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import SortConfig
from .sorter import PhotoSorter, validate_config


class SorterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Сортировка фото шкафов НКУ")
        self.geometry("920x620")

        self.photos_var = tk.StringVar()
        self.excel_var = tk.StringVar()
        self.objects_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="copy")

        self._build_ui()

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}
        main = ttk.Frame(self)
        main.pack(fill="both", expand=True)

        self._path_row(main, 0, "Папка с фото", self.photos_var, is_file=False)
        self._path_row(main, 1, "Excel-реестр (.xlsx)", self.excel_var, is_file=True)
        self._path_row(main, 2, "Корень папки «Объекты»", self.objects_var, is_file=False)
        self._path_row(main, 3, "Папка результата", self.output_var, is_file=False)

        mode_frame = ttk.LabelFrame(main, text="Режим работы")
        mode_frame.grid(row=4, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Radiobutton(mode_frame, text="Копировать (copy)", variable=self.mode_var, value="copy").pack(side="left", padx=10)
        ttk.Radiobutton(mode_frame, text="Перемещать (move)", variable=self.mode_var, value="move").pack(side="left", padx=10)

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=5, column=0, columnspan=3, sticky="ew", **pad)

        ttk.Button(btn_frame, text="Запустить сортировку", command=self._start).pack(side="left", padx=6)
        ttk.Button(btn_frame, text="Открыть папку результата", command=self._open_output).pack(side="left", padx=6)

        self.log = tk.Text(main, wrap="word", height=20)
        self.log.grid(row=6, column=0, columnspan=3, sticky="nsew", **pad)

        main.columnconfigure(1, weight=1)
        main.rowconfigure(6, weight=1)

    def _path_row(self, parent: ttk.Frame, row: int, title: str, var: tk.StringVar, is_file: bool) -> None:
        ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", padx=8, pady=6)
        choose_cmd = (lambda: self._choose_file(var)) if is_file else (lambda: self._choose_dir(var))
        ttk.Button(
            parent,
            text="Выбрать",
            command=choose_cmd,
        ).grid(row=row, column=2, sticky="ew", padx=8, pady=6)

    def _choose_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory()
        if path:
            var.set(path)

    def _choose_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")])
        if path:
            var.set(path)

    def _append_log(self, line: str) -> None:
        self.log.insert("end", line + "\n")
        self.log.see("end")

    def _config_from_ui(self) -> SortConfig:
        return SortConfig(
            photos_dir=Path(self.photos_var.get()),
            excel_file=Path(self.excel_var.get()),
            objects_root=Path(self.objects_var.get()),
            output_dir=Path(self.output_var.get()),
            mode=self.mode_var.get(),
        )

    def _start(self) -> None:
        config = self._config_from_ui()
        try:
            validate_config(config)
        except Exception as exc:
            messagebox.showerror("Ошибка конфигурации", str(exc))
            return

        self._append_log("Запуск обработки...")

        def worker() -> None:
            try:
                sorter = PhotoSorter(config, logger=self._append_log)
                sorter.run()
                self.after(0, lambda: messagebox.showinfo("Готово", "Сортировка завершена"))
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror("Ошибка", str(exc)))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _open_output(self) -> None:
        output = self.output_var.get().strip()
        if not output:
            messagebox.showwarning("Внимание", "Сначала укажите папку результата")
            return
        try:
            os.startfile(output)  # type: ignore[attr-defined]
        except Exception as exc:
            messagebox.showerror("Ошибка", f"Не удалось открыть папку: {exc}")


def run_gui() -> None:
    app = SorterApp()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
