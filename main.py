import cv2
import numpy as np

font = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread("RectangleTest1.png", cv2.IMREAD_GRAYSCALE)
_, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
contours,hierachy  = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# while len(contours) > 0:
for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    # approx = cv2.approxPolyDP(cnt, 4, True)
    # x = approx.ravel()[0]
    # y = approx.ravel()[1]
    if len(approx) == 4:
        cv2.drawContours(img, [cnt], 0, (0, 0, 0), 10)
        cv2.imshow("shapes", img)
        # print(approx)
        # cv2.drawContours(img,[cnt],0, (255,255,255), 10)
        # contours, hierachy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
# contours,hierachy  = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1]
#     if len(approx) == 4:
#         cv2.drawContours(img,[cnt],0, (255,255,255), 10)
#
# _, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
# contours,hierachy  = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1]
#     if len(approx) == 4:
#         cv2.drawContours(img,[cnt],0, (255,255,255), 10)


# words_to_remove = ['on', 'you', 'crazy', 'diamond']
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#
# image = cv2.imread("1.png")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# inverted_thresh = 255 - thresh
# dilate = cv2.dilate(inverted_thresh, kernel, iterations=4)
#
# cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     ROI = thresh[y:y+h, x:x+w]
#     data = pytesseract.image_to_string(ROI, lang='eng',config='--psm 6').lower()
#     if data in words_to_remove:
#         image[y:y+h, x:x+w] = [255,255,255]


cv2.imshow("shapes", img)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()