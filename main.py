import os
import json

from ocr.aadhaar_ocr import AadhaarOCR


def get_latest_warped():

    capture_dir = "captures"

    files = [
        os.path.join(capture_dir, f)
        for f in os.listdir(capture_dir)
        if f.startswith("warped_")
    ]

    if not files:
        raise Exception("No warped images found")

    return max(files, key=os.path.getctime)


def main():

    image_path = get_latest_warped()

    print(f"\nUsing image: {image_path}\n")

    ocr = AadhaarOCR()

    result = ocr.process(image_path)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()