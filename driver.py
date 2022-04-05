import cv2
from Rectangle.rectangle import RectangleRemover
from OCR.OCRscan import OCRscan
from LineDetection.findcountmethod import LineDetection

if __name__ == '__main__':
    # This line allows CNTL-C in the terminal to kill the program
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    img = cv2.imread("OCR/SimpleAsteroid.png", cv2.IMREAD_GRAYSCALE)
    ocr = OCRscan()
    img, words, text_list = ocr.scan(img)
    app = RectangleRemover(img, words, text_list)
    rectangles, image_no_rec = app.remove_rectangle()
    lineDetection = LineDetection()
    lineDetection.detectLine(rectangles, image_no_rec)
    # sys.exit(app.exec())