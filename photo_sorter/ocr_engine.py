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
class SerialCandidate:
    serial: str
    priority: int
    source: str
    context: str
    reason: str
    rejected_reason: str | None = None
    from_marker: bool = False


@dataclass
class LabelScanResult:
    is_label: bool
    serial: str | None
    confidence: float
    debug_text: str
    candidates: list[SerialCandidate]
    chosen_reason: str


class LabelRecognizer:
    def __init__(self, config: SortConfig) -> None:
        self.config = config
        self.reader = easyocr.Reader(["ru", "en"], gpu=False)

        keywords = [re.escape(x.lower()) for x in config.label_keywords]
        self.keyword_patterns = [re.compile(k) for k in keywords]

        self.serial_marker = re.compile(r"(завод\.?\s*№|зав\.?\s*№|заводской\s*№)", re.IGNORECASE)
        self.serial_number_pattern = re.compile(r"\b\d{6,12}\b")

        self.reject_tu = re.compile(r"\bту\b", re.IGNORECASE)
        self.reject_year_or_spec = re.compile(
            r"(год\s*выпуска|степень\s*защиты|номинальн|\bip\b|\beac\b)", re.IGNORECASE
        )

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

    def _regions(self, img: np.ndarray) -> list[tuple[np.ndarray, str]]:
        h, w = img.shape[:2]
        out: list[tuple[np.ndarray, str]] = []
        for idx, (rx, ry, rw, rh) in enumerate(self.config.region_specs):
            x, y = int(w * rx), int(h * ry)
            ww, hh = int(w * rw), int(h * rh)
            crop = img[y : min(y + hh, h), x : min(x + ww, w)]
            if crop.size:
                out.append((crop, f"region_{idx}"))
        return out

    def _keyword_score(self, text: str) -> int:
        low = text.lower()
        return sum(1 for p in self.keyword_patterns if p.search(low))

    def _read_text(self, img: np.ndarray) -> str:
        result = self.reader.readtext(img, detail=0, paragraph=False)
        return "\n".join(str(x) for x in result)

    def _scan_qr_payload(self, img: np.ndarray) -> str | None:
        if not self.config.include_qr_scan:
            return None
        decoded = decode(img)
        for item in decoded:
            try:
                payload = item.data.decode("utf-8", errors="ignore")
            except Exception:
                continue
            if payload.strip():
                return payload
        return None

    def extract_serial_candidates(self, text_blocks: list[dict[str, str]]) -> list[SerialCandidate]:
        """Извлекает кандидаты серийного номера из OCR-блоков c приоритетами и контекстом."""
        candidates: list[SerialCandidate] = []

        for block in text_blocks:
            source = block.get("source", "unknown")
            text = block.get("text", "")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            for idx, line in enumerate(lines):
                line_low = line.lower()
                line_candidates = self.serial_number_pattern.findall(line)
                if not line_candidates:
                    continue

                prev_line = lines[idx - 1] if idx > 0 else ""
                next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
                near_text = f"{prev_line} {line} {next_line}".strip()
                near_low = near_text.lower()

                marker_hit = bool(self.serial_marker.search(near_low))
                tu_hit = bool(self.reject_tu.search(near_low))
                spec_hit = bool(self.reject_year_or_spec.search(near_low))

                for serial in line_candidates:
                    priority = 20
                    reason = "candidate_plain"
                    rejected_reason: str | None = None

                    if marker_hit:
                        priority = 100
                        reason = "marker_nearby"

                    if tu_hit:
                        rejected_reason = "rejected_as_tu_number"
                        priority = -10
                    elif spec_hit:
                        rejected_reason = "rejected_as_year_or_spec"
                        priority = -5

                    candidates.append(
                        SerialCandidate(
                            serial=serial,
                            priority=priority,
                            source=source,
                            context=near_text[:200],
                            reason=reason,
                            rejected_reason=rejected_reason,
                            from_marker=marker_hit,
                        )
                    )

        # дедуп по serial, берем лучший вариант по priority
        best_by_serial: dict[str, SerialCandidate] = {}
        for cand in candidates:
            prev = best_by_serial.get(cand.serial)
            if prev is None or cand.priority > prev.priority:
                best_by_serial[cand.serial] = cand

        return sorted(best_by_serial.values(), key=lambda c: c.priority, reverse=True)

    def _choose_serial_candidate(
        self,
        candidates: list[SerialCandidate],
        registry_serials: set[str] | None,
    ) -> tuple[SerialCandidate | None, str]:
        valid = [c for c in candidates if c.rejected_reason is None]
        if not valid:
            return None, ""

        marker_candidates = [c for c in valid if c.from_marker]
        non_marker = [c for c in valid if not c.from_marker]

        chosen: SerialCandidate | None = marker_candidates[0] if marker_candidates else valid[0]
        chosen_reason = "chosen_by_label_marker" if marker_candidates else "chosen_by_priority"

        if self.config.verify_by_registry and registry_serials:
            if chosen and chosen.serial not in registry_serials:
                for cand in marker_candidates + non_marker:
                    if cand.serial in registry_serials:
                        chosen = cand
                        chosen_reason = "chosen_by_registry_match"
                        break

        return chosen, chosen_reason

    def scan_file(self, image_path: Path, registry_serials: set[str] | None = None) -> LabelScanResult:
        try:
            source = load_rgb_image(image_path)
        except Exception as exc:
            return LabelScanResult(False, None, 0.0, f"ERROR: {exc}", [], "")

        qr_payload = self._scan_qr_payload(source)
        if qr_payload:
            qr_candidates = self.extract_serial_candidates([{"source": "qr", "text": qr_payload}])
            qr_valid = [c for c in qr_candidates if c.rejected_reason is None]
            if qr_valid:
                chosen, reason = self._choose_serial_candidate(qr_valid, registry_serials)
                if chosen:
                    return LabelScanResult(True, chosen.serial, 1.0, "QR", qr_candidates, f"qr_{reason or 'highest'}")

        text_blocks: list[dict[str, str]] = []

        # 1) быстрый шаг: OCR полного кадра без поворота
        full_text = self._read_text(source)
        text_blocks.append({"source": "full_0", "text": full_text})
        quick_candidates = self.extract_serial_candidates(text_blocks)
        quick_choice, quick_reason = self._choose_serial_candidate(quick_candidates, registry_serials)

        if quick_choice:
            is_label = self._keyword_score(full_text) >= 2 or quick_choice.from_marker
            if is_label:
                return LabelScanResult(
                    True,
                    quick_choice.serial,
                    0.95,
                    "full_0",
                    quick_candidates,
                    quick_reason,
                )

        # 2) повороты полного кадра при отсутствии номера
        for angle in (90, 180, 270):
            rot = self._rotate(source, angle)
            rot_text = self._read_text(rot)
            text_blocks.append({"source": f"full_{angle}", "text": rot_text})

        candidates_after_rot = self.extract_serial_candidates(text_blocks)
        rotated_choice, rotated_reason = self._choose_serial_candidate(candidates_after_rot, registry_serials)
        merged_text = "\n".join(block["text"] for block in text_blocks)
        if rotated_choice:
            is_label = self._keyword_score(merged_text) >= 2 or rotated_choice.from_marker
            if is_label:
                return LabelScanResult(
                    True,
                    rotated_choice.serial,
                    0.9,
                    "full_rotations",
                    candidates_after_rot,
                    rotated_reason,
                )

        # 3) тяжелый шаг: многозонный OCR только если не нашли serial ранее
        if not self.config.fast_mode:
            for angle in (0, 90, 180, 270):
                rotated = self._rotate(source, angle)
                for region_img, region_name in self._regions(rotated):
                    text_blocks.append({"source": f"{region_name}_a{angle}", "text": self._read_text(region_img)})

        all_text = "\n".join(block["text"] for block in text_blocks)
        all_candidates = self.extract_serial_candidates(text_blocks)
        chosen, chosen_reason = self._choose_serial_candidate(all_candidates, registry_serials)
        is_label = self._keyword_score(all_text) >= 2

        if chosen and (is_label or chosen.from_marker):
            return LabelScanResult(True, chosen.serial, 0.8, "full+regions", all_candidates, chosen_reason)

        return LabelScanResult(is_label, None, 0.4 if is_label else 0.0, "no_serial", all_candidates, "")
