from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import easyocr
import numpy as np
from pyzbar.pyzbar import decode

from .config import SortConfig
from .image_io import load_rgb_image


@dataclass
class LabelScanResult:
    is_label: bool
    serial: str | None
    confidence: float
    debug_text: str


class LabelRecognizer:
    def __init__(self, config: SortConfig) -> None:
        self.config = config
        # Русский + английский для EAC и технических обозначений.
        self.reader = easyocr.Reader(["ru", "en"], gpu=False)

        keywords = [re.escape(x.lower()) for x in config.label_keywords]
        self.keyword_patterns = [re.compile(k) for k in keywords]
        self.serial_hint = re.compile(r"(завод\.?\s*№|зав\.?\s*№|заводской\s*№)", re.IGNORECASE)
        self.serial_number_pattern = re.compile(r"\b\d{6,12}\b")

    def _rotate(self, img: np.ndarray, angle: int) -> np.ndarray:
        if angle == 0:
            return img
        if angle == 90:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(img, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def _regions(self, img: np.ndarray) -> list[np.ndarray]:
        h, w = img.shape[:2]
        out: list[np.ndarray] = []
        for rx, ry, rw, rh in self.config.region_specs:
            x, y = int(w * rx), int(h * ry)
            ww, hh = int(w * rw), int(h * rh)
            crop = img[y : min(y + hh, h), x : min(x + ww, w)]
            if crop.size:
                out.append(crop)
        return out

    def _extract_serial_from_text(self, text: str) -> tuple[str | None, float]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        best_serial: str | None = None
        best_score = 0.0

        for i, line in enumerate(lines):
            low = line.lower()
            if self.serial_hint.search(low):
                joined = line
                if i + 1 < len(lines):
                    joined += " " + lines[i + 1]
                candidates = self.serial_number_pattern.findall(joined)
                if candidates:
                    candidate = max(candidates, key=len)
                    return candidate, 1.0

        fallback = self.serial_number_pattern.findall(text)
        for candidate in fallback:
            score = 0.4
            if len(candidate) >= 8:
                score += 0.2
            if candidate.startswith("0"):
                score += 0.1
            if score > best_score:
                best_score = score
                best_serial = candidate

        return best_serial, best_score

    def _keyword_score(self, text: str) -> int:
        low = text.lower()
        hits = sum(1 for p in self.keyword_patterns if p.search(low))
        return hits

    def _read_text(self, img: np.ndarray) -> str:
        result = self.reader.readtext(img, detail=0, paragraph=False)
        return "\n".join(str(x) for x in result)

    def _scan_qr(self, img: np.ndarray) -> str | None:
        if not self.config.include_qr_scan:
            return None
        decoded = decode(img)
        for item in decoded:
            try:
                payload = item.data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            serials = self.serial_number_pattern.findall(payload)
            if serials:
                return serials[0]
        return None

    def scan_file(self, image_path: Path) -> LabelScanResult:
        try:
            source = load_rgb_image(image_path)
        except Exception as exc:
            return LabelScanResult(False, None, 0.0, f"ERROR: {exc}")

        best = LabelScanResult(False, None, 0.0, "")

        qr_serial = self._scan_qr(source)
        if qr_serial:
            return LabelScanResult(True, qr_serial, 1.0, "QR")

        for angle in (0, 90, 180, 270):
            rotated = self._rotate(source, angle)
            for idx, region in enumerate(self._regions(rotated)):
                text = self._read_text(region)
                kw_score = self._keyword_score(text)
                serial, serial_conf = self._extract_serial_from_text(text)

                label_conf = min(1.0, kw_score / 4.0)
                total_conf = (0.7 * label_conf) + (0.3 * serial_conf)

                is_label = kw_score >= 2 and serial is not None
                debug = f"angle={angle};region={idx};kw={kw_score};serial={serial};text={text[:180]}"

                if is_label and total_conf >= best.confidence:
                    best = LabelScanResult(True, serial, total_conf, debug)
                elif total_conf > best.confidence:
                    best = LabelScanResult(False, serial, total_conf, debug)

        return best
