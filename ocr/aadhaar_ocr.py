import cv2
import pytesseract
import re


class AadhaarOCR:

    def __init__(self):

        pytesseract.pytesseract.tesseract_cmd = (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )

    def preprocess(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        gray = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        return gray

    def extract_aadhaar(self, text):

        text = text.replace("\n", " ")

        match = re.search(
            r"\b\d{4}\s?\d{4}\s?\d{4}\b",
            text
        )

        if match:
            number = match.group()

            number = re.sub(
                r"(\d{4})(\d{4})(\d{4})",
                r"\1 \2 \3",
                number.replace(" ", "")
            )

            return number

        return None

    def extract_dob(self, text):

        match = re.search(
            r"\d{2}/\d{2}/\d{4}",
            text
        )

        if match:
            return match.group()

        return None

    def extract_gender(self, text):

        text = text.upper()

        if "MALE" in text:
            return "Male"

        if "FEMALE" in text:
            return "Female"

        return None

    def extract_mobile(self, text):

        match = re.search(
            r"\b[6-9]\d{9}\b",
            text
        )

        if match:
            return match.group()

        return None

    def extract_name(self, text):

        lines = text.split("\n")

        ignore = [
            "government",
            "india",
            "dob",
            "male",
            "female",
            "mobile",
            "aadhaar"
        ]

        candidates = []

        for line in lines:

            line = line.strip()

            if len(line) < 5:
                continue

            if any(
                word in line.lower()
                for word in ignore
            ):
                continue

            if re.search(r"\d", line):
                continue

            candidates.append(line)

        if len(candidates) > 0:

            return max(
                candidates,
                key=len
            )

        return None

    def process(self, image_path):

        image = cv2.imread(image_path)

        print("Image path:", image_path)
        print("Image loaded:", image is not None)

        if image is None:
            raise Exception(f"Could not load image: {image_path}")

        # processed = self.preprocess(image)

        # cv2.imwrite("debug_preprocessed.jpg", processed)

        # text = pytesseract.image_to_string(
        #     processed,
        #     lang="eng"
        # )

        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        print(text)
        
        print("RAW OCR:")
        print(repr(text))

        return {
            "name": self.extract_name(text),
            "dob": self.extract_dob(text),
            "gender": self.extract_gender(text),
            "mobile": self.extract_mobile(text),
            "aadhaar_number": self.extract_aadhaar(text),
            "raw_text": text
        }