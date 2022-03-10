import collections
import cv2
import pytesseract
import numpy as np
from pytesseract import Output

from Rectangle.rectangle import RectangleRemover
from OCR.OCRscan import OCRscan

if __name__ == '__main__':
    # This line allows CNTL-C in the terminal to kill the program
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    img = cv2.imread("OCR/example.png", cv2.IMREAD_GRAYSCALE)
    ocr = OCRscan()
    img, words = ocr.scan(img)
    app = RectangleRemover(img, words)
    app.remove_rectangle()
    # sys.exit(app.exec())