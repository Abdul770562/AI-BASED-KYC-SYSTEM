"""
Production-level Aadhaar Card OCR
Handles poor lighting, skew, blur, glare, and low-res photos.
"""

from email.mime import text

import cv2
import numpy as np
import pytesseract
import re
import os
import platform
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from PIL import Image
import torch

from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel
)


# ── Tesseract path (auto-detected) ───────────────────────────────────────────

def _find_tesseract() -> Optional[str]:
    if platform.system() == "Windows":
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
    return None  # Let pytesseract find it on PATH (Linux/macOS)


_tess = _find_tesseract()
if _tess:
    pytesseract.pytesseract.tesseract_cmd = _tess


# ── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class AadhaarResult:
    name:           Optional[str] = None
    dob:            Optional[str] = None
    gender:         Optional[str] = None
    mobile:         Optional[str] = None
    aadhaar_number: Optional[str] = None
    confidence:     float = 0.0          # 0–1, how much was extracted
    raw_text:       str = ""
    strategy_used:  str = ""
    warnings:       list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "dob":            self.dob,
            "gender":         self.gender,
            "mobile":         self.mobile,
            "aadhaar_number": self.aadhaar_number,
            "confidence":     round(self.confidence, 2),
            "raw_text":       self.raw_text,
            "strategy_used":  self.strategy_used,
            "warnings":       self.warnings,
        }


# ── Image pre-processing strategies ──────────────────────────────────────────

class ImageProcessor:
    """
    Multiple preprocessing pipelines ordered from least to most aggressive.
    Each returns a list of BGR images to try with Tesseract.
    """

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 2:
            return img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _resize_if_small(img: np.ndarray, min_width: int = 1000) -> np.ndarray:
        h, w = img.shape[:2]
        if w < min_width:
            scale = min_width / w
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def _deskew(gray: np.ndarray) -> np.ndarray:
        """Correct rotation using Hough lines or min-area-rect on text blobs."""
        coords = np.column_stack(np.where(gray < 128))
        if len(coords) < 100:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated

    @staticmethod
    def _remove_glare(img: np.ndarray) -> np.ndarray:
        """Suppress bright specular reflections that whiteout text."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # Mask overexposed regions (L > 230)
        glare_mask = l > 230
        l[glare_mask] = np.mean(l[~glare_mask]).astype(np.uint8)
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _clahe(gray: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _sharpen(gray: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(gray, -1, kernel)

    @staticmethod
    def _denoise(gray: np.ndarray) -> np.ndarray:
        return cv2.fastNlMeansDenoising(gray, h=10)

    # ── pipeline variants ─────────────────────────────────────────────────────

    def strategy_clean(self, img: np.ndarray) -> list[np.ndarray]:
        """Best for well-lit, straight photos."""
        img = self._resize_if_small(img)
        gray = self._to_gray(img)
        gray = self._deskew(gray)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [gray, binary]

    def strategy_adaptive(self, img: np.ndarray) -> list[np.ndarray]:
        """Best for uneven lighting / shadows."""
        img = self._resize_if_small(img)
        gray = self._to_gray(img)
        gray = self._deskew(gray)
        gray = self._clahe(gray)
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10)
        return [gray, adaptive]

    def strategy_denoised(self, img: np.ndarray) -> list[np.ndarray]:
        """Best for grainy / low-light shots."""
        img = self._resize_if_small(img, min_width=1200)
        gray = self._to_gray(img)
        gray = self._denoise(gray)
        gray = self._clahe(gray)
        gray = self._deskew(gray)
        sharpened = self._sharpen(gray)
        _, binary = cv2.threshold(
            sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [gray, sharpened, binary]

    def strategy_glare(self, img: np.ndarray) -> list[np.ndarray]:
        """Best for photos with flash / reflections."""
        img = self._resize_if_small(img)
        img = self._remove_glare(img)
        gray = self._to_gray(img)
        gray = self._deskew(gray)
        gray = self._clahe(gray)
        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 25, 8)
        return [gray, adaptive]

    def strategy_pil_enhance(self, img: np.ndarray) -> list[np.ndarray]:
        """PIL-based contrast/sharpness boost — often rescues blurry scans."""
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil = pil.resize(
            (max(pil.width, 1000), max(pil.height, 600)),
            Image.LANCZOS,
        )
        pil = ImageEnhance.Contrast(pil).enhance(2.0)
        pil = ImageEnhance.Sharpness(pil).enhance(2.5)
        pil = pil.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
        arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        gray = self._to_gray(arr)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [gray, binary]

    def all_variants(self, img: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Return (strategy_name, processed_image) pairs for every variant."""
        results = []
        strategies = [
            ("clean",        self.strategy_clean),
            ("adaptive",     self.strategy_adaptive),
            ("denoised",     self.strategy_denoised),
            ("glare",        self.strategy_glare),
            ("pil_enhance",  self.strategy_pil_enhance),
        ]
        for name, fn in strategies:
            try:
                for i, variant in enumerate(fn(img)):
                    results.append((f"{name}_{i}", variant))
            except Exception:
                pass
        return results


# ── OCR runner ────────────────────────────────────────────────────────────────

# class OCREngine:
#     # Tesseract configs to try (order matters — faster/simpler first)
#     CONFIGS = [
#         "--oem 3 --psm 4",   # single column of text
#         "--oem 3 --psm 6",   # uniform block of text
#         "--oem 3 --psm 11",  # sparse text
#         "--oem 1 --psm 6",   # LSTM only
#     ]

#     def run(self, gray: np.ndarray) -> str:
#         best = ""
#         for cfg in self.CONFIGS:
#             try:
#                 text = pytesseract.image_to_string(gray, lang="eng", config=cfg)
#                 if len(text.strip()) > len(best.strip()):
#                     best = text
#             except Exception:
#                 pass
#         return best

class DonutEngine:

    def __init__(self):

        self.processor = DonutProcessor.from_pretrained(
            "naver-clova-ix/donut-base"
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            "naver-clova-ix/donut-base"
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)

    def run(self, image):

        if len(image.shape) == 2:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )
        else:
            image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        pil_image = Image.fromarray(image)

        pixel_values = self.processor(
            pil_image,
            return_tensors="pt"
        ).pixel_values

        pixel_values = pixel_values.to(
            self.device
        )

        task_prompt = "<s>"

        decoder_input_ids = self.processor.tokenizer(
            task_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).input_ids.to(self.device)

        outputs = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=1024,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=3
        )

        text = self.processor.batch_decode(
            outputs,
            skip_special_tokens=True
        )[0]

        return text


# ── Field extractors ─────────────────────────────────────────────────────────

class FieldExtractor:

    # Common OCR mis-reads of digits
    _DIGIT_FIX = str.maketrans({
        "O": "0", "o": "0",
        "I": "1", "l": "1",
        "S": "5", "s": "5",
        "Z": "2", "z": "2",
        "B": "8",
        "G": "6",
    })

    def _fix_digits(self, s: str) -> str:
        return s.translate(self._DIGIT_FIX)

    # ── Aadhaar number ────────────────────────────────────────────────────────

    def extract_aadhaar(self, text: str) -> Optional[str]:
        clean = self._fix_digits(text).replace("\n", " ")

        # Try to find 12 consecutive digits (with optional spaces every 4)
        patterns = [
            r"\b(\d{4})[\s\-]?(\d{4})[\s\-]?(\d{4})\b",
            r"\b(\d{4})\s(\d{4})\s(\d{4})\b",
        ]
        for pat in patterns:
            m = re.search(pat, clean)
            if m:
                num = "".join(m.groups())
                # Basic Verhoeff / length check
                # Real Aadhaar numbers don't start with 0 or 1;
                # we still accept them during extraction — caller can validate.
                if len(num) == 12:
                    return f"{num[:4]} {num[4:8]} {num[8:]}"
        return None

    # ── Date of birth ─────────────────────────────────────────────────────────

    def extract_dob(self, text: str) -> Optional[str]:
        clean = self._fix_digits(text)

        # Strategy 1: anchor on DOB label — intentionally loose because OCR
        # often mangles "जन्म तारीख/DOB:" into garbage like "wes mte/DOB".
        # We only need the label to survive partially to anchor the date.
        labelled = re.search(
            r"(?:DOB|D\.O\.B|Birth|tarikh)[^0-9]{0,20}"
            r"(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})",
            clean, re.IGNORECASE
        )
        if labelled:
            d, mo, y = labelled.groups()
            return f"{d}/{mo}/{y}"

        # Strategy 2: bare date anywhere in text
        patterns = [
            r"\b(\d{2})[/\-\.](\d{2})[/\-\.](\d{4})\b",  # DD/MM/YYYY
            r"\b(\d{4})[/\-\.](\d{2})[/\-\.](\d{2})\b",  # YYYY/MM/DD
            r"Year of Birth[^\d]*(\d{4})",                        # masked cards
        ]
        for pat in patterns:
            m = re.search(pat, clean, re.IGNORECASE)
            if m:
                groups = m.groups()
                if len(groups) == 1:
                    return f"YYYY/{groups[0]}"
                d, mo, y = groups
                if len(d) == 4:
                    d, y = y, d
                return f"{d}/{mo}/{y}"
        return None

    # ── Gender ────────────────────────────────────────────────────────────────

    def extract_gender(self, text: str) -> Optional[str]:
        upper = text.upper()
        # Handle OCR variants: MALE / M A L E / MAIE
        if re.search(r"\bFE?M[AI]LE?\b", upper):
            return "Female"
        if re.search(r"\bM[AI]LE?\b", upper):
            return "Male"
        # Hindi transliteration fallback
        if "पुरुष" in text:
            return "Male"
        if "महिला" in text:
            return "Female"
        return None

    # ── Mobile ────────────────────────────────────────────────────────────────

    def extract_mobile(self, text: str) -> Optional[str]:
        clean = self._fix_digits(text)
        m = re.search(r"\b([6-9]\d{9})\b", clean)
        return m.group(1) if m else None

    # ── Name ─────────────────────────────────────────────────────────────────

    _IGNORE_WORDS = {
        "government", "india", "dob", "male", "female", "mobile",
        "aadhaar", "aadhar", "unique", "identification", "authority",
        "enrolment", "enrollment", "address", "download", "digitally",
        "signed", "issue", "date", "year", "birth", "card", "uid",
        "help", "centre", "resident", "valid", "proof",
    }

    def extract_name(self, text: str) -> Optional[str]:
        lines = [l.strip() for l in text.split("\n")]
        candidates = []

        for line in lines:
            if len(line) < 4 or len(line) > 70:
                continue
            lower = line.lower()
            if any(w in lower for w in self._IGNORE_WORDS):
                continue
            if re.search(r"\d", line):
                continue
            # Must look like a name: only ASCII letters, spaces, dots, hyphens.
            # Non-ASCII (Hindi/Devanagari) lines are rejected here so we only
            # return the romanised English name printed on the card.
            if not re.match(r"^[A-Za-z][A-Za-z\s\.\-]{2,}$", line):
                continue
            words = line.split()
            # Need at least 2 words — single-word lines are usually noise
            if len(words) < 2:
                continue
            candidates.append(line)

        if not candidates:
            return None

        def name_score(s: str) -> tuple:
            words = s.split()
            # Count title-cased words (real name words start with capital)
            titled = sum(1 for w in words if w and w[0].isupper())
            # Penalise ALL-CAPS lines (those tend to be headings/labels)
            is_allcaps = s == s.upper()
            return (titled, -int(is_allcaps), len(s))

        return max(candidates, key=name_score)


# ── Scoring / result selection ────────────────────────────────────────────────

def _score(result: AadhaarResult) -> float:
    """Score 0–1 based on how many fields were successfully extracted."""
    weights = {
        "aadhaar_number": 0.40,
        "name":           0.25,
        "dob":            0.20,
        "gender":         0.10,
        "mobile":         0.05,
    }
    score = 0.0
    for field, w in weights.items():
        if getattr(result, field) is not None:
            score += w
    return score


# ── Main AadhaarOCR class ─────────────────────────────────────────────────────

class AadhaarOCR:
    """
    Production Aadhaar OCR.

    Usage
    -----
    ocr = AadhaarOCR()
    result = ocr.process("photo.jpg")   # returns AadhaarResult
    print(result.to_dict())
    """

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.processor = ImageProcessor()
        self.engine = DonutEngine()
        self.extractor = FieldExtractor()

    # ── public API ────────────────────────────────────────────────────────────

    def process(self, image_path: str) -> AadhaarResult:
        image = self._load(image_path)

        # Try every (strategy, config) combination and keep the best result
        all_variants = self.processor.all_variants(image)
        best: Optional[AadhaarResult] = None

        for strategy_name, variant in all_variants:
            text = self.engine.run(variant)
            if not text.strip():
                continue

            print("\nDONUT OUTPUT:\n")
            print(text)
            print("\n------------------\n")

            result = self._extract_all(text, strategy_name)

            if self.debug:
                print(f"[{strategy_name}] score={_score(result):.2f}  "
                      f"aadhaar={result.aadhaar_number}  name={result.name}")

            if best is None or _score(result) > _score(best):
                best = result
                if _score(best) >= 0.95:  # early exit if near-perfect
                    break

        if best is None:
            best = AadhaarResult(warnings=["No text could be extracted from image."])
        else:
            best.confidence = _score(best)
            if best.confidence < 0.4:
                best.warnings.append(
                    "Low confidence — image quality may be too poor. "
                    "Try better lighting or a higher-resolution photo."
                )

        return best

    def process_image(self, image: np.ndarray) -> AadhaarResult:
        """Accept an already-loaded BGR numpy array (e.g. from cv2.VideoCapture)."""
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, image)
        result = self.process(tmp.name)
        os.unlink(tmp.name)
        return result

    # ── internals ─────────────────────────────────────────────────────────────

    def _load(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {path}")
        return img

    def _extract_all(self, text: str, strategy: str) -> AadhaarResult:
        e = self.extractor
        return AadhaarResult(
            name=           e.extract_name(text),
            dob=            e.extract_dob(text),
            gender=         e.extract_gender(text),
            mobile=         e.extract_mobile(text),
            aadhaar_number= e.extract_aadhaar(text),
            raw_text=       text,
            strategy_used=  strategy,
        )


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json

    if len(sys.argv) < 2:
        print("Usage: python aadhaar_ocr.py <image_path> [--debug]")
        sys.exit(1)

    debug = "--debug" in sys.argv
    ocr = AadhaarOCR(debug=debug)
    result = ocr.process(sys.argv[1])

    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))