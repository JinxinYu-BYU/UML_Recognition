import collections
import cv2
import pytesseract
import numpy as np
from pytesseract import Output

from Rectangle.rectangle import RectangleRemover
from OCR.OCRscan import OCRscan
# from OCR.text import Text

if __name__ == '__main__':
    # This line allows CNTL-C in the terminal to kill the program
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    img = cv2.imread("OCR/example.png", cv2.IMREAD_GRAYSCALE)
    ocr = OCRscan()
    img, words, text_list = ocr.scan(img)
    app = RectangleRemover(img, words, text_list)
    app.remove_rectangle()
    # print(Text(1, 1, 1, 1, "hello").getText())
    # sys.exit(app.exec())