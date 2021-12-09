import cv2
import numpy as np
import matplotlib.pyplot as plt
import queue




class Rectangle:
    def __init__(self, points):
        if len(points) == 8:
            self.topLeft = [points[0], points[1]]
            self.bottomLeft = [points[2], points[3]]
            self.bottomRight = [points[4], points[5]]
            self.topRight = [points[6], points[7]]

    @classmethod
    def fromRectangles(cls, bottomRec, topRec):

        return cls

    def getLeftX(self):
        if (self.topLeft[0] - self.bottomLeft[0]) < 20:
            return (self.topLeft[0] + self.bottomLeft[0]) / 2

    def getRightX(self):
        if (self.topRight[0] - self.bottomRight[0]) < 20:
            return (self.topRight[0] + self.bottomRight[0]) / 2

    def getTopY(self):
        if (self.topLeft[1] - self.topRight[1]) < 20:
            return (self.topLeft[1] + self.topRight[1]) / 2

    def getBottomY(self):
        if (self.bottomLeft[1] - self.bottomRight[1]) < 20:
            return (self.bottomLeft[1] + self.bottomRight[1]) / 2

    def getWidth(self):
        return self.getRightX() - self.getLeftX()

class RecMax(Rectangle):
    def __init__(self, p_queue):
        pass






def group_rectangles():
    classes = []
    leftXMap = {}
    for rectangle_key in widthMap:
        rectangle_queue = widthMap.get(rectangle_key)
        min = rectangle_queue.get()[1]
        y_queue = queue.PriorityQueue()
        y_queue.put((min.getBottomY(), min))
        leftXMap[(min.getLeftX(), rectangle_key)] = y_queue

        while not rectangle_queue.empty():
            next = rectangle_queue.get()[1]
            if next.getLeftX() - min.getLeftX() < 3:
                if next.getBottomY() is not None:
                    leftXMap.get((min.getLeftX(), rectangle_key)).put((next.getBottomY(), next))
            else:
               min = next
               y_queue = queue.PriorityQueue()
               y_queue.put((min.getBottomY(), min))
               leftXMap[(min.getLeftX(), rectangle_key)] = y_queue

    for key in leftXMap:
        left_queue = leftXMap.get(key)
        min = left_queue.get()[1]
        if left_queue.qsize() > 1:
            while not left_queue.empty():
                next = left_queue.get()[1]
                if next.getBottomY() - min.getTopY() < 5:
                    points = [next.topLeft[0], next.topLeft[1], min.bottomLeft[0], min.bottomLeft[1],
                              min.bottomRight[0], min.bottomRight[1], next.topRight[0], next.topRight[1]]
                    min = Rectangle(points)
                else:
                    classes.append(min)
                    min = next
        else:
            classes.append(min)
    return classes




font = cv2.FONT_HERSHEY_COMPLEX

img = cv2.imread("./Rectangle/RectangleTest1.png", cv2.IMREAD_GRAYSCALE)
_, threshold = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
contours, hierachy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rectangles = []
plt.figure()
widthMap = {}
a = 0
for cnt in contours:
    print("index: ", "\n")
    # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    approx = cv2.approxPolyDP(cnt, 4, True)
    # if len(approx) == 4 or len(approx) == 6:
    if len(approx) == 4:
        cv2.drawContours(img, [cnt], 0, (255, 255, 255), 10)

        # print(approx)
        n = approx.ravel()
        rectangle = Rectangle(n)
        width = rectangle.getWidth() - rectangle.getWidth() % 5
        if width in widthMap:
            widthMap.get(width).put((rectangle.getLeftX(), rectangle))
        else:
            x_queue = queue.PriorityQueue()
            x_queue.put((rectangle.getLeftX(), rectangle))
            widthMap[width] = x_queue
        i = 0
        # cv2.putText(img, str(a), (n[0] + 6, n[1] + 6),
        #             font, 10, (255, 0, 0))
        print(n)
        for j in n:
            if (i % 2 == 0):
                x = n[i]
                y = n[i + 1]
                # String containing the co-ordinates.
                string = str(x) + " " + str(y)
                # text on remaining co-ordinates.
                cv2.putText(img, string, (x, y),
                            font, 0.5, (0, 255, 0))
                plt.scatter(x, y)
            i = i + 1
    # a = a + 1

classes = group_rectangles()
# for rec in classes:
#     cv2.rectangle(img, rec.topLeft, rec.bottomRight, (0, 0, 0), 10)


# percent by which the image is resized
scale_percent = 65

# calculate the 50 percent of original dimensions
WidndowWidth = int(img.shape[1] * 100 / 100)
height = int(img.shape[0] * scale_percent / 100)

# dsize
dsize = (WidndowWidth, height)

# resize image
output = cv2.resize(img, dsize)
threshold = cv2.resize(threshold,dsize)

cv2.imshow("shapes", output)
cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()

plt.imshow(img)
plt.show()
