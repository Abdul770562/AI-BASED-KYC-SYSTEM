import cv2
import pytesseract
import easyocr
import torch

from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)



class OCRBenchmark:

    def __init__(self):

        # EasyOCR
        self.easy_reader = easyocr.Reader(
            ["en"],
            gpu=torch.cuda.is_available()
        )

        # TrOCR
        self.trocr_processor = (
            TrOCRProcessor.from_pretrained(
                "microsoft/trocr-base-printed"
            )
        )

        self.trocr_model = (
            VisionEncoderDecoderModel.from_pretrained(
                "microsoft/trocr-base-printed"
            )
        )

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.trocr_model.to(self.device)

    # -------------------------------
    # TESSERACT
    # -------------------------------

    def run_tesseract(self, image):

        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--oem 3 --psm 6"
        )

        return text

    # -------------------------------
    # EASYOCR
    # -------------------------------

    def run_easyocr(self, image):

        results = self.easy_reader.readtext(
            image,
            detail=0
        )

        return "\n".join(results)

    # -------------------------------
    # TrOCR
    # -------------------------------

    def run_trocr(self, image):



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

        h, w = image.shape[:2]

        # name_crop = image[
        #     int(h * 0.30):int(h * 0.55),
        #     int(w * 0.10):int(w * 0.95)
        # ]

        name_crop = image[
            int(h * 0.22):int(h * 0.42),
            int(w * 0.08):int(w * 0.95)
        ]

        cv2.imwrite(
            "debug_name_crop.jpg",
            name_crop
        )

        pil_image = Image.fromarray(
            cv2.cvtColor(
                name_crop,
                cv2.COLOR_BGR2RGB
            )
        )

        pixel_values = (
            self.trocr_processor(
                images=pil_image,
                return_tensors="pt"
            ).pixel_values
        )

        pixel_values = pixel_values.to(
            self.device
        )

        generated_ids = (
            self.trocr_model.generate(
                pixel_values,
                max_length=512
            )
        )

        text = self.trocr_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text

    # -------------------------------
    # BENCHMARK
    # -------------------------------

    def benchmark(self, image_path):


        image = cv2.imread(image_path)

        print("\n")
        print("=" * 80)
        print("TESSERACT")
        print("=" * 80)

        tess_text = self.run_tesseract(
            image
        )

        print(tess_text)

        print("\n")
        print("=" * 80)
        print("EASYOCR")
        print("=" * 80)

        easy_text = self.run_easyocr(
            image
        )

        print(easy_text)

        print("\n")
        print("=" * 80)
        print("TrOCR")
        print("=" * 80)

        trocr_text = self.run_trocr(
            image
        )

        print(trocr_text)

        return {
            "tesseract": tess_text,
            "easyocr": easy_text,
            "trocr": trocr_text
        }


if __name__ == "__main__":

    image_path = (
        r"captures\warped_20260611_195222.jpg"
    )

    benchmark = OCRBenchmark()

    benchmark.benchmark(image_path)