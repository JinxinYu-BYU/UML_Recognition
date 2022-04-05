import collections
import cv2
import pytesseract
import numpy as np
from pytesseract import Output
import OCR.OCRtext
from json import JSONEncoder


class OCRscan:
    def __init__( self):
        pass

    def scan(self, img):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # Adding custom options
        custom_config = r'-c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890+- -c tessedit_char_blacklist=_|\[]'
        # h, w, c = img.shape
        # img = get_grayscale(img)
        # img = thresholding(img)
        # text = pytesseract.image_to_string(img, config=custom_config, lang='eng')
        results = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
        i, j = 0, 1
        out = collections.defaultdict(list)
        text_list = []
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        text = results["text"][i]

        text = str(text).strip()
        while j < len(results["text"]):
            x1 = results["left"][j]
            y1 = results["top"][j]
            w1 = results["width"][j]
            h1 = results["height"][j]
            text1 = results["text"][j]
            if abs(y1 - y) <= 15 and abs(x + w - x1) < 20:
                text1 = results["text"][j]
                text1 = str(text1).strip()
                text = text + " " + text1
                w = w1 + abs(x1 - x)
                j += 1
            else:
                text_list.append(OCR.OCRtext.OCRtext(x, y, w, h, text))
                out["left"].append(x)
                out["top"].append(y)
                out["width"].append(w)
                out["height"].append(h)
                out["text"].append(text)
                i = j
                x = results["left"][i]
                y = results["top"][i]
                w = results["width"][i]
                h = results["height"][i]
                text = results["text"][i]
                j += 1
        if j == len(results["text"]):
            text = results["text"][i]
            text = str(text).strip()
            i = j
        for i in range(0, len(out["text"])):
            # We can then extract the bounding box coordinates
                # of the text region from  the current result
            x = out["left"][i]
            y = out["top"][i]
            w = out["width"][i]
            h = out["height"][i]
            text = out["text"][i]
            # print(text)
            text = str(text).strip()
            if len(text) > 2:
                # print("Text: {}, Len: {}, Top Left: {}, {}, BttmRight: {}, {}".format(text, len(text), x, y, x+w, y+h))
                # print("")
                cv2.rectangle(img,
                              (x-5, y-5),
                              (x + w + 5, y + h + 5),
                              (255, 255, 255), -1)
                # cv2.putText(img,
                #             text,
                #             (x, y+h),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             0.5, (255, 0, 0), 1)

        self.displayImg(img)
        return img, out, text_list


    def displayImg(self, img):
        img = self.resizeImg(img)
        cv2.imshow('img', img)
        cv2.waitKey(0)

    def resizeImg(self, img):
        scale_percent = 50
        # calculate the 50 percent of original dimensions
        WidndowWidth = int(img.shape[1] * 60 / 100)
        height = int(img.shape[0] * scale_percent / 100)
        # dsize
        dsize = (WidndowWidth, height)
        # resize image
        img = cv2.resize(img, dsize)
        return img

    def get_grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(self, image):
        return cv2.medianBlur(image, 5)

    # thresholding
    def thresholding(self, image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # dilation
    def dilate(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

    # erosion
    def erode(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # opening - erosion followed by dilation
    def opening(self, image):
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # canny edge detection
    def canny(self, image):
        return cv2.Canny(image, 100, 200)
