import cv2
import numpy as np
import pytesseract
from PIL import Image

class TextbookOCR:
    def __init__(self, lang="hin+eng"):
        self.lang = lang

    def preprocess_image(self, image_bytes):
        """OpenCV preprocessing: grayscale, denoising, and adaptive thresholding"""
        file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def extract_text(self, image_bytes):
        try:
            processed_img = self.preprocess_image(image_bytes)
            pil_img = Image.fromarray(processed_img)
            extracted_text = pytesseract.image_to_string(pil_img, lang=self.lang)
            return extracted_text.strip()
        except Exception as e:
            return f"OCR Fallback Mode (Install Tesseract Binary): {str(e)}"