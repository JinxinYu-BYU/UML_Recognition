import collections

import cv2
import numpy as np
import queue


class Rectangle:
    def __init__(self, points):
        self.values = []
        if len(points) == 8:
            i = 0
            set = []
            while i < len(points):
                t = (points[i], points[i + 1])
                set.append(t)
                i += 2
            set.sort(key=lambda x: x[0])
            if set[0][1] < set[1][1]:
                self.topLeft = set[0]
                self.bottomLeft = set[1]
            else:
                self.topLeft = set[1]
                self.bottomLeft = set[0]

            if set[2][1] < set[3][1]:
                self.topRight = set[2]
                self.bottomRight = set[3]
            else:
                self.topRight = set[3]
                self.bottomRight = set[2]

    @classmethod
    def fromRectangles(cls, bottomRec, topRec):
        return cls

    def getLeftX(self):
        # if (self.topLeft[0] - self.bottomLeft[0]) < 20:
        return (self.topLeft[0] + self.bottomLeft[0]) / 2

    def getRightX(self):
        # if (self.topRight[0] - self.bottomRight[0]) < 20:
        return (self.topRight[0] + self.bottomRight[0]) / 2

    def getTopY(self):
        # if (self.topLeft[1] - self.topRight[1]) < 20:
        return (self.topLeft[1] + self.topRight[1]) / 2

    def getBottomY(self):
        # if (self.bottomLeft[1] - self.bottomRight[1]) < 20:
        return (self.bottomLeft[1] + self.bottomRight[1]) / 2

    def getWidth(self):
        return self.getRightX() - self.getLeftX()

    def getValues(self):
        return self.values

    def addValues(self, values):
        self.values.append(values)

class RecMax(Rectangle):
    def __init__(self, p_queue):
        pass

class RectangleRemover:
    def __init__(self, img, words):
        self.img = img
        self.words = words

    def group_rectangles(self, widthMap):
        classes = []
        keys = list(widthMap.keys())
        keys.sort()
        i = 0
        while i < len(keys):
            # print("widthmap")
            rectangle_key = keys[i]
            leftXMap = {}
            rectangle_list = []
            temp_list_1 = widthMap[rectangle_key].copy()
            temp_list_2 = widthMap[rectangle_key - 5].copy()
            temp_list_3 = widthMap[rectangle_key + 5].copy()
            # rectangle_list.sort(key=lambda x:x[0])
            for rec in temp_list_1: rectangle_list.append((1, rec))
            for rec in temp_list_2: rectangle_list.append((2, rec))
            for rec in temp_list_3: rectangle_list.append((3, rec))

            for tuple in rectangle_list:
                # print("rectanglelist")
                leftXIndex = tuple[1].getLeftX() - tuple[1].getLeftX() % 5
                if leftXIndex in leftXMap:
                    leftXMap.get(leftXIndex).append(tuple)
                else:
                    # x_queue = queue.PriorityQueue()
                    leftX_list = []
                    leftX_list.append(tuple)
                    leftXMap[leftXIndex] = leftX_list

            low_bar = -2
            for key in leftXMap:
                # print("leftXMap")
                topY_list = leftXMap.get(key, []).copy()
                topY_list += leftXMap.get(key + 5, []).copy()
                topY_list += leftXMap.get(key - 5, []).copy()
                topY_list.sort(key=lambda x: x[1].getTopY())

                curr = 0
                curr_rectangle_tuple = topY_list[0]
                while curr <= len(topY_list) - 1:
                    # print("TopYlist")
                    next = curr  ##TODO: check next bounds
                    while topY_list[next][1].getTopY() - curr_rectangle_tuple[1].getBottomY() < -2 and next + 1 < len(
                            topY_list):  ## overlap
                        next += 1

                    if -2 < topY_list[next][1].getTopY() - curr_rectangle_tuple[1].getBottomY() <= 5:
                        points = [curr_rectangle_tuple[1].topLeft[0], curr_rectangle_tuple[1].topLeft[1],
                                  topY_list[next][1].bottomLeft[0], topY_list[next][1].bottomLeft[1],
                                  topY_list[next][1].bottomRight[0], topY_list[next][1].bottomRight[1],
                                  curr_rectangle_tuple[1].topRight[0], curr_rectangle_tuple[1].topRight[1]]
                        curr_rectangle = Rectangle(points)
                        topY_list.pop(next)
                        curr_rectangle_tuple = (curr_rectangle_tuple[0], curr_rectangle)
                        topY_list[curr] = curr_rectangle_tuple
                        continue

                    if topY_list[next][1].getTopY() - curr_rectangle_tuple[
                        1].getBottomY() > 5:  ## gap too big, impossible to merge
                        curr += 1
                        if curr < len(topY_list):
                            curr_rectangle_tuple = topY_list[curr]
                    # if curr == next == len(topY_list) - 1 :
                    if (topY_list[next][1].getTopY() - curr_rectangle_tuple[1].getBottomY() < -2 and next == len(
                            topY_list) - 1):  ##overlaping rectangle
                        break

                for item in topY_list:
                    if item[0] != 3:
                        classes.append(item[1])

            i += 1
        return classes

    def remove_rectangle(self):
        font = cv2.FONT_HERSHEY_COMPLEX
        # img = cv2.imread("RectangleTest1.png", cv2.IMREAD_GRAYSCALE)
        _, threshold = cv2.threshold(self.img, 200, 255, cv2.THRESH_BINARY)
        contours, hierachy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = []
        widthMap = collections.defaultdict(list)
        a = 0
        for cnt in contours:
            # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            approx = cv2.approxPolyDP(cnt, 4, True)
            # if len(approx) == 4 or len(approx) == 6:
            if len(approx) == 4:
                cv2.drawContours(self.img, [cnt], 0, (255, 255, 255), 10)

                # print(approx)
                n = approx.ravel()
                rectangle = Rectangle(n)
                width = rectangle.getWidth() - rectangle.getWidth() % 5
                if width in widthMap:
                    ## TODO: check if we need to keep leftX in the tuple
                    widthMap.get(width).append(rectangle)
                else:
                    # x_queue = queue.PriorityQueue()
                    # TODO: check the pq implementation
                    x_list = []
                    x_list.append(rectangle)
                    widthMap[width] = x_list
                i = 0
                # cv2.putText(img, str(a), (n[0] + 6, n[1] + 6),
                #             font, 10, (255, 0, 0))
                # print(n)
                for j in n:
                    if (i % 2 == 0):
                        x = n[i]
                        y = n[i + 1]
                        # String containing the co-ordinates.
                        string = str(x) + " " + str(y)
                        # text on remaining co-ordinates.
                        cv2.putText(self.img, string, (x, y),
                                    font, 0.5, (0, 255, 0))
                        cv2.putText(threshold, string, (x, y),
                                    font, 0.5, (0, 255, 0))
                        # plt.scatter(x, y)
                    i = i + 1
            # a = a + 1
        ##TODO: put all text into the rectangles
        print(widthMap.values())
        for recList in widthMap.values():
            # for each word
            # if word's width < rec's width && word's top > rec's top && word's bottom < rec's bottom
            # add text values
            for rec in recList:
                for i in range(len(self.words["text"])):
                    if self.words["left"][i] >= rec.getLeftX() and (self.words["left"][i] + self.words["width"][i]) <= rec.getRightX() \
                            and self.words["top"][i] >= rec.getTopY() and (self.words["top"][i] + self.words["height"][i]) <= rec.getBottomY():
                        if (not self.words["text"][i].isspace()) and (len(self.words["text"][i])) > 0:
                            print("added: " + self.words["text"][i])
                            print("length: " + str(len(self.words["text"][i])))
                            rec.addValues((self.words["text"][i], self.words["left"][i], self.words["top"][i], self.words["width"][i], self.words["height"][i]))

        print("before testing")
        print(self.words["text"])
        # testing
        for recList in widthMap.values():
            for rec in recList:
                print("topleft: {}, bottomright: {}, values: {}".format(rec.topLeft, rec.bottomRight, rec.getValues()))

        classes = self.group_rectangles(widthMap)
        for rec in classes:
            cv2.rectangle(self.img, rec.topLeft, rec.bottomRight, (0, 0, 0), 10)
            for i in range(len(rec.getValues())):
                cv2.putText(self.img,
                            rec.getValues()[i][0],
                            (rec.getValues()[i][1], rec.getValues()[i][2]+rec.getValues()[i][4]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (51, 153, 255), 2)



        self.displayImage(threshold)

    def displayImage(self, threshold):
        # percent by which the image is resized
        scale_percent = 36

        # calculate the 50 percent of original dimensions
        WidndowWidth = int(self.img.shape[1] * scale_percent / 100)
        height = int(self.img.shape[0] * scale_percent / 100)

        # dsize
        dsize = (WidndowWidth, height)

        # resize image
        output = cv2.resize(self.img, dsize)
        threshold = cv2.resize(threshold, dsize)

        cv2.imshow("shapes", output)
        cv2.imshow("Threshold", threshold)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # This line allows CNTL-C in the terminal to kill the program
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    img = cv2.imread("RectangleTest1.png", cv2.IMREAD_GRAYSCALE)
    app = RectangleRemover(img)
    app.remove_rectagnle()
    # sys.exit(app.exec())