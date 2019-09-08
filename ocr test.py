import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


import pytesseract
from PIL import Image

value=Image.open("beans.jpg")
text = pytesseract.image_to_string(value, config='')
print("text present in images:",text)