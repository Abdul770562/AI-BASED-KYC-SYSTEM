"""
Aadhaar OCR pipeline entry-point.
Picks the latest warped capture, runs OCR, validates, and prints JSON output.
"""

import os
import sys
import json

from ocr.aadhaar_ocr import AadhaarOCR, AadhaarResult
from validators.aadhaar_validator import AadhaarValidator


# ── Image selection ───────────────────────────────────────────────────────────

def get_latest_warped(capture_dir: str = "captures") -> str:
    """Return the most-recently created warped_*.* file in capture_dir."""
    if not os.path.isdir(capture_dir):
        raise FileNotFoundError(
            f"Capture directory not found: '{capture_dir}'"
        )

    files = [
        os.path.join(capture_dir, f)
        for f in os.listdir(capture_dir)
        if f.startswith("warped_")
    ]

    if not files:
        raise FileNotFoundError(
            f"No warped images found in '{capture_dir}'. "
            "Run the capture step first."
        )

    return max(files, key=os.path.getctime)


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=4, ensure_ascii=False))


def _build_output(result: AadhaarResult, validation: dict) -> dict:
    """Combine OCR result and validation into a single output dict."""
    return {
        "extracted_data": result.to_dict(),
        "validation":     validation,
        "summary": {
            "confidence":    result.confidence,
            "strategy_used": result.strategy_used,
            "warnings":      result.warnings,
            "status": (
                "ok"      if result.confidence >= 0.8  else
                "partial" if result.confidence >= 0.4  else
                "failed"
            ),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Resolve image path ─────────────────────────────────────────────────
    try:
        image_path = get_latest_warped()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing: {image_path}\n")

    # ── 2. Run OCR ────────────────────────────────────────────────────────────
    debug = "--debug" in sys.argv
    ocr = AadhaarOCR(debug=debug)

    try:
        result: AadhaarResult = ocr.process(image_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] OCR failed unexpectedly: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── 3. Validate ───────────────────────────────────────────────────────────
    validator = AadhaarValidator()

    # AadhaarValidator.validate() expects a plain dict — pass result.to_dict()
    # so this file stays compatible with any validator implementation.
    try:
        validation_result = validator.validate(result.to_dict())
    except Exception as exc:
        validation_result = {"error": str(exc), "valid": False}

    # ── 4. Print output ───────────────────────────────────────────────────────
    output = _build_output(result, validation_result)
    _print_json(output)

    # ── 5. Exit code reflects extraction quality ──────────────────────────────
    if result.confidence < 0.4:
        print(
            "\n[WARN] Low confidence extraction. "
            "Retake the photo in better lighting.",
            file=sys.stderr,
        )
        sys.exit(2)   # partial / failed — caller can detect and retry


if __name__ == "__main__":
    main()