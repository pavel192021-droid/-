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
    from_target_zone: bool = False
    in_registry: bool = False


@dataclass
class LabelScanResult:
    is_label: bool
    serial: str | None
    confidence: float
    debug_text: str
    candidates: list[SerialCandidate]
    chosen_reason: str
    full_frame_candidates: list[str]
    region_candidates: list[str]
    target_zone_candidates: list[str]
    globally_rejected_candidates: list[str]
    registry_matches: list[str]
    final_decision_reason: str
    result_full_frame: str
    result_region_scan: str
    result_targeted_bottom_right: str


class LabelRecognizer:
    def __init__(self, config: SortConfig) -> None:
        self.config = config
        self.reader = easyocr.Reader(["ru", "en"], gpu=False)

        keywords = [re.escape(x.lower()) for x in config.label_keywords]
        self.keyword_patterns = [re.compile(k) for k in keywords]

        # Поддержка OCR-искажений маркера "зав. №"
        self.serial_marker = re.compile(
            r"((з|3|s)\s*[аa]\s*[вvbб]\.?|завод\.?)\s*(№|n|no|n°|nо)?",
            re.IGNORECASE,
        )
        self.serial_number_pattern = re.compile(r"\b\d{6,12}\b")

        self.reject_tu = re.compile(r"\bту\b", re.IGNORECASE)
        self.reject_year_or_spec = re.compile(
            r"(год\s*выпуска|степень\s*защиты|номинальн|\bip\b|\beac\b)",
            re.IGNORECASE,
        )
        self.reject_spec_number = re.compile(r"(\d{2}\.\d{2}\.\d{2}|\b\d{4}\b-\d{2,})")

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

    def _read_text_boxes(self, img: np.ndarray):
        return self.reader.readtext(img, detail=1, paragraph=False)

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

    def _preprocess_for_targeted(self, crop: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        up = cv2.resize(gray, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_CUBIC)
        contrast = cv2.convertScaleAbs(up, alpha=1.7, beta=8)
        thr = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            7,
        )
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(thr, -1, kernel)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)

    def _targeted_zones(self, img: np.ndarray) -> list[tuple[np.ndarray, str]]:
        h, w = img.shape[:2]
        zones: list[tuple[np.ndarray, str]] = []

        zones.append((img[h // 2 : h, w // 2 : w], "target_bottom_right_quarter"))
        zones.append((img[(2 * h) // 3 : h, (2 * w) // 3 : w], "target_bottom_right_third"))

        # Поиск зон рядом с маркером "зав"
        try:
            box_results = self._read_text_boxes(img)
            for idx, item in enumerate(box_results[:20]):
                box, text, _ = item
                if not self.serial_marker.search(str(text).lower()):
                    continue
                xs = [int(p[0]) for p in box]
                ys = [int(p[1]) for p in box]
                x1, x2 = max(0, min(xs)), min(w, max(xs) + int(0.45 * w))
                y1, y2 = max(0, min(ys) - int(0.04 * h)), min(h, max(ys) + int(0.12 * h))
                crop = img[y1:y2, x1:x2]
                if crop.size:
                    zones.append((crop, f"target_near_marker_{idx}"))
        except Exception:
            pass

        return [(self._preprocess_for_targeted(c), name) for c, name in zones if c.size]

    def extract_serial_candidates(self, text_blocks: list[dict[str, str]]) -> list[SerialCandidate]:
        candidates: list[SerialCandidate] = []

        for block in text_blocks:
            source = block.get("source", "unknown")
            text = block.get("text", "")
            is_target = source.startswith("target_")
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            for idx, line in enumerate(lines):
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
                spec_number_hit = bool(self.reject_spec_number.search(near_low))

                for serial in line_candidates:
                    priority = 30
                    reason = "candidate_plain"
                    rejected_reason: str | None = None

                    if marker_hit:
                        priority = 1000
                        reason = "marker_nearby"
                    elif is_target:
                        priority = 850
                        reason = "target_zone_candidate"

                    if tu_hit:
                        rejected_reason = "rejected_as_tu_number"
                        priority = -100
                    elif spec_hit:
                        rejected_reason = "rejected_as_year_or_spec"
                        priority = -90
                    elif spec_number_hit:
                        rejected_reason = "rejected_as_spec_number"
                        priority = -80

                    candidates.append(
                        SerialCandidate(
                            serial=serial,
                            priority=priority,
                            source=source,
                            context=near_text[:220],
                            reason=reason,
                            rejected_reason=rejected_reason,
                            from_marker=marker_hit,
                            from_target_zone=is_target,
                        )
                    )

        # Дедуп: оставляем лучший вариант для каждого serial
        best_by_serial: dict[str, SerialCandidate] = {}
        for cand in candidates:
            prev = best_by_serial.get(cand.serial)
            if prev is None or cand.priority > prev.priority:
                best_by_serial[cand.serial] = cand

        return sorted(best_by_serial.values(), key=lambda c: c.priority, reverse=True)

    @staticmethod
    def _apply_global_rejection(candidates: list[SerialCandidate]) -> tuple[list[SerialCandidate], set[str]]:
        rejected_serials = {
            c.serial
            for c in candidates
            if c.rejected_reason in {"rejected_as_tu_number", "rejected_as_year_or_spec", "rejected_as_spec_number"}
        }
        filtered = [c for c in candidates if c.serial not in rejected_serials]
        return filtered, rejected_serials

    def _choose_serial_candidate(
        self,
        candidates: list[SerialCandidate],
        registry_serials: set[str] | None,
    ) -> tuple[SerialCandidate | None, str, list[str]]:
        if registry_serials:
            for cand in candidates:
                cand.in_registry = cand.serial in registry_serials

        registry_matches = [c.serial for c in candidates if c.in_registry]

        marker = [c for c in candidates if c.from_marker]
        targeted = [c for c in candidates if c.from_target_zone and not c.from_marker]
        registry_only = [c for c in candidates if c.in_registry and not c.from_marker and not c.from_target_zone]
        others = [c for c in candidates if c not in marker and c not in targeted and c not in registry_only]

        # A: маркер + реестр
        for cand in marker:
            if cand.in_registry:
                return cand, "chosen_by_label_marker", registry_matches

        if self.config.verify_by_registry:
            # В строгом режиме выбираем только номера, которые есть в реестре.
            for group, reason in (
                (marker, "chosen_by_registry_match"),
                (targeted, "chosen_by_registry_match"),
                (registry_only, "chosen_by_registry_match"),
                (others, "chosen_by_registry_match"),
            ):
                for cand in group:
                    if cand.in_registry:
                        return cand, reason, registry_matches
            return None, "no_registry_match", registry_matches

        # Нестрогий режим: A -> B -> C -> D
        if marker:
            return marker[0], "chosen_by_label_marker", registry_matches
        if targeted:
            return targeted[0], "chosen_by_target_zone", registry_matches
        if registry_only:
            return registry_only[0], "chosen_by_registry_match", registry_matches
        if others:
            return others[0], "chosen_by_priority", registry_matches

        return None, "no_candidates", registry_matches

    @staticmethod
    def _candidate_strings(cands: list[SerialCandidate]) -> list[str]:
        return [f"{c.serial}|{c.source}|p={c.priority}|{c.rejected_reason or 'ok'}" for c in cands]

    def scan_file(self, image_path: Path, registry_serials: set[str] | None = None) -> LabelScanResult:
        try:
            source = load_rgb_image(image_path)
        except Exception as exc:
            return LabelScanResult(False, None, 0.0, f"ERROR: {exc}", [], "", [], [], [], [], [], "error", "", "", "")

        full_blocks: list[dict[str, str]] = []
        region_blocks: list[dict[str, str]] = []
        target_blocks: list[dict[str, str]] = []

        # QR с максимальным приоритетом
        qr_payload = self._scan_qr_payload(source)
        if qr_payload:
            target_blocks.append({"source": "target_qr", "text": qr_payload})

        # full frame
        full_blocks.append({"source": "full_0", "text": self._read_text(source)})
        base_text = full_blocks[0]["text"]
        label_likely = self._keyword_score(base_text) >= 2

        # повороты full frame
        for angle in (90, 180, 270):
            rot = self._rotate(source, angle)
            full_blocks.append({"source": f"full_{angle}", "text": self._read_text(rot)})

        all_full_text = "\n".join(b["text"] for b in full_blocks)
        label_likely = label_likely or self._keyword_score(all_full_text) >= 2

        # targeted OCR обязателен для фото-шильдика;
        # а в fast_mode обязателен если serial после full frame не найден.
        full_candidates = self.extract_serial_candidates(full_blocks)
        full_filtered, full_rejected = self._apply_global_rejection(full_candidates)
        prelim_choice, _, _ = self._choose_serial_candidate(full_filtered, registry_serials)
        need_targeted = label_likely or prelim_choice is None

        if need_targeted:
            for zone_img, zone_name in self._targeted_zones(source):
                target_blocks.append({"source": zone_name, "text": self._read_text(zone_img)})

        # region OCR (тяжелый) только не в fast_mode
        if not self.config.fast_mode:
            for angle in (0, 90, 180, 270):
                rotated = self._rotate(source, angle)
                for region_img, region_name in self._regions(rotated):
                    region_blocks.append({"source": f"{region_name}_a{angle}", "text": self._read_text(region_img)})

        all_blocks = full_blocks + target_blocks + region_blocks
        all_candidates = self.extract_serial_candidates(all_blocks)
        filtered, globally_rejected = self._apply_global_rejection(all_candidates)

        chosen, chosen_reason, registry_matches = self._choose_serial_candidate(filtered, registry_serials)

        # итоговый статус в strict режиме: если нет совпадений с реестром, serial не выбираем
        if self.config.verify_by_registry and chosen is None:
            final_reason = "no_registry_match"
            chosen_serial = None
        else:
            final_reason = chosen_reason
            chosen_serial = chosen.serial if chosen else None

        label_by_marker = any(c.from_marker for c in filtered)
        is_label = label_likely or label_by_marker

        # Сводки для GUI/логов
        full_strings = self._candidate_strings(self.extract_serial_candidates(full_blocks))
        region_strings = self._candidate_strings(self.extract_serial_candidates(region_blocks))
        target_strings = self._candidate_strings(self.extract_serial_candidates(target_blocks))

        return LabelScanResult(
            is_label=is_label,
            serial=chosen_serial,
            confidence=0.95 if chosen_serial else (0.5 if is_label else 0.0),
            debug_text=f"final_reason={final_reason}",
            candidates=filtered,
            chosen_reason=chosen_reason,
            full_frame_candidates=full_strings,
            region_candidates=region_strings,
            target_zone_candidates=target_strings,
            globally_rejected_candidates=sorted(globally_rejected),
            registry_matches=sorted(set(registry_matches)),
            final_decision_reason=final_reason,
            result_full_frame=full_strings[0] if full_strings else "none",
            result_region_scan=region_strings[0] if region_strings else "none",
            result_targeted_bottom_right=target_strings[0] if target_strings else "none",
        )
