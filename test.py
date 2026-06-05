# import openbharatocr
# import pytesseract

# pytesseract.pytesseract.tesseract_cmd = (
#     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

# front_result = openbharatocr.front_aadhaar("captures\\warped_20260605_210515.jpg")

# print(front_result)

import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

img = cv2.imread(
    "captures\\warped_20260605_210515.jpg"
)

text = pytesseract.image_to_string(img)

print(text)