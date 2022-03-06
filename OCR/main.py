import cv2
import pytesseract
import numpy as np
from pytesseract import Output

img = cv2.imread('test2_big.png')


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Adding custom options
# custom_config = r'--oem 1 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyz1234567890+- -c tessedit_char_blacklist=_><|\[] --psm 3'
custom_config = r'-c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890+- -c tessedit_char_blacklist=_|\[]'
# custom_config = r'--oem 1 -c tessedit_char_blacklist=abcdefghijklmnopqrstuvwxyz1234567890 --psm 1'



# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


# canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)


h, w, c = img.shape
# img = get_grayscale(img)
# img = thresholding(img)
text = pytesseract.image_to_string(img, config=custom_config, lang='eng')
# print(text)
#
# boxes = pytesseract.image_to_boxes(img, config=custom_config, lang='eng')
# img = canny(img)


# for b in boxes.splitlines():
#     b = b.split(' ')
#     print(b, "\n")
#     if b[0].isalpha() or b[0].isnumeric() and b[0] != "~":
#     # if b[0] == "~":
#     #     img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 0, 0), 1)
#         img = cv2.rectangle(img, (int(b[1]) - 4, h - int(b[2]) + 4), (int(b[3]) + 4, h - int(b[4]) - 4), (255, 255, 255), -1)

results = pytesseract.image_to_data(img, config=custom_config, output_type=Output.DICT)
i, j = 0, 1
# while i < len(results["text"]):
for i in range(0, len(results["text"])):
    # We can then extract the bounding box coordinates
    # of the text region from  the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    text = results["text"][i]
    # print(text)
    text = str(text).strip()

    # while j < len(results["text"]):
    #     x1 = results["left"][j]
    #     y1 = results["top"][j]
    #     w1 = results["width"][j]
    #     h1 = results["height"][j]
    #
    #     if abs(y1 - y) < 2 and abs(x + w - x1) < 5:
    #         text1 = results["text"][j]
    #         text1 = str(text1).strip()
    #         text = text + text1
    #         w = w1 + abs(x1 - x)
    #         j += 1
    #     else:
    #         i = j
    #         x = results["left"][i]
    #         y = results["top"][i]
    #         w = results["width"][i]
    #         h = results["height"][i]
    #         text = results["text"][i]
    #         j += 1
    #
    # if j == len(results["text"]):
    #     text = results["text"][i]
    #     text = str(text).strip()
    #     i = j



    # We then strip out non-ASCII text so we can
    # draw the text on the image We will be using
    # OpenCV, then draw a bounding box around the
    # text along with the text itself
    if len(text) > 2:
        print("Text: {}, Len: {}, Top Left: {}, {}, BttmRight: {}, {}".format(text, len(text), x, y, x+w, y+h))
        print("")

        cv2.rectangle(img,
                      (x, y),
                      (x + w, y + h),
                      (255, 255, 255), -1)
        cv2.putText(img,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 3)
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# boxes = pytesseract.image_to_boxes(img, lang='eng')
# print(boxes)
# for b in boxes.splitlines():
#     b = b.split(' ')
#     if b[0].isalpha():
#         img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (255, 255, 0), 1)

scale_percent = 50

# calculate the 50 percent of original dimensions
WidndowWidth = int(img.shape[1] * 40 / 100)
height = int(img.shape[0] * scale_percent / 100)

# dsize
dsize = (WidndowWidth, height)

# resize image
img = cv2.resize(img, dsize)

cv2.imshow('img', img)
cv2.waitKey(0)

# 999149348
