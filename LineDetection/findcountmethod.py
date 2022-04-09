import cv2
# import matplotlib.pyplot as plt
import numpy as np
import collections
import json

from Rectangle.rectangle import Rectangle, Encoder

class LineDetection:

    def detectLine(self, classes, image):
        # image = cv2.imread('line2.png', cv2.IMREAD_COLOR)
        pr_image = cv2.imread("OCR/SimpleAsteroid.png", cv2.COLOR_RGB2GRAY)
        print(pr_image.shape)
        print(image.shape)
        gray_image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE) # cv2.COLOR_RGB2GRAY

        # canny edge detection
        canny_image = cv2.Canny(gray_image, 100, 200)


        contours, hierachy = cv2.findContours(canny_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        ## TODO : This is the output, each list inside this dictionary represents a line(includes the endpoints of one line)
        # outCnt = collections.defaultdict(list)
        outCnt = []
        index = 0
        while index in range(len(contours)):
            lineSet = []
            cnt = contours[index]
            blank_image = np.zeros([image.shape[0], image.shape[1]], np.uint8)
            for i in cnt:
                blank_image[i[0][1], i[0][0]] = 255

            # Create a copy of the mask for points processing:
            groupsMask = blank_image.copy()

            # Set kernel (structuring element) size:
            kernelSize = 3
            # Set operation iterations:
            opIterations = 3
            # Get the structuring element:
            maxKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
            # Perform dilate:
            groupsMask = cv2.morphologyEx(groupsMask, cv2.MORPH_DILATE, maxKernel, None, None, opIterations, cv2.BORDER_REFLECT101)


            # Set the centroids Dictionary:
            centroidsDictionary = {}

            # Get centroids on the end points mask:
            totalComponents, output, stats, centroids = cv2.connectedComponentsWithStats(blank_image, connectivity=4)

            # Count the blob labels with this:
            labelCounter = 1

            # Loop through the centroids, skipping the background (0):
            for c in range(1, len(centroids), 1):

                # Get the current centroids:
                cx = int(centroids[c][0])
                cy = int(centroids[c][1])

                # Get the pixel value on the groups mask:
                pixelValue = groupsMask[cy, cx]

                # If new value (255) there's no entry in the dictionary
                # Process a new key and value:
                if pixelValue == 255:

                    # New key and values-> Centroid and Point Count:
                    centroidsDictionary[labelCounter] = (cx, cy, 1)

                    # Flood fill at centroid:
                    cv2.floodFill(groupsMask, mask=None, seedPoint=(cx, cy), newVal=labelCounter)
                    labelCounter += 1

                # Else, the label already exists and we must accumulate the
                # centroid and its count:
                else:

                    # Get Value:
                    (accumCx, accumCy, blobCount) = centroidsDictionary[pixelValue]

                    # Accumulate value:
                    accumCx = accumCx + cx
                    accumCy = accumCy + cy
                    blobCount += 1

                    # Update dictionary entry:
                    centroidsDictionary[pixelValue] = (accumCx, accumCy, blobCount)


            # Loop trough the dictionary and get the final centroid values:

            for k in centroidsDictionary:
                # Get the value of the current key:
                (cx, cy, count) = centroidsDictionary[k]
                # Process combined points:
                if count != 1:
                    cx = int(cx/count)
                    cy = int(cy/count)
                # Draw circle at the centroid
                if index < 4:
                    cv2.circle(pr_image, (cx, cy), 5, (0, 0, 255), -1)
                lineSet.append((int(cx), int(cy)))
                cv2.rectangle(image, (cx-33, cy-33),(cx+33, cy+33),(0, 255, 255), 2 )
                cropped_image = image[cx: cx + 66, cy:cy + 66]
                print(recognize(cropped_image))
            outCnt.append(lineSet)
            index += 1

        # corners = cv2.goodFeaturesToTrack(gray_image,25,0.01,10)
        # corners = cv2.cornerHarris(gray_image,2,3,0.04)
        # corners = np.int0(corners)
        # for i in corners:
        #     x,y = i.ravel()
        #     cv2.circle(image,(x,y),3,255,-1)

        print("outCnt.values:", outCnt)
        self.match(outCnt, classes, pr_image)
        cv2.imshow('image', image)
        cv2.imshow('blimage', blank_image)
        #
        # # Exiting the window if 'q' is pressed on the keyboard.
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()









        ## TODO: build a relationship class that contains two classes (means they are connected, doesn't care what relationship they have right now)
        ## if one line connects two different classes, build a relationship instance for them and add the relationship to a list
        ## return a list of relationships

    def isInRec(self, rec, point):
        thresh = 15
        if rec.getTopY() - thresh < point[1] < rec.getTopY() + thresh and rec.getLeftX() - thresh < point[0] < rec.getRightX() + thresh:
            return "Top"
        elif rec.getBottomY() - 20 < point[1] < rec.getBottomY() + 20 and rec.getLeftX() - thresh < point[0] < rec.getRightX() + thresh:
            return "Bottom"
        elif rec.getTopY() - thresh < point[1] < rec.getBottomY() + thresh and rec.getLeftX() - thresh < point[0] < rec.getLeftX() + thresh:
            return "Left"
        elif rec.getTopY() - thresh < point[1] < rec.getBottomY() + thresh and rec.getRightX() - thresh < point[0] < rec.getRightX() + thresh:
            return "Right"
        else:
            return None

    def match(self, outCnt, classes, pr_image):
        allRelationships = []
        # [(Ship.id, Top), (Bullet.id: bottom)]
        ## TODO: use two for loops to match each class with each point in outCnt
        for line in outCnt:
            relationship = Relationship()
            for endPoint in line:
                for rec in classes:
                    side = self.isInRec(rec, endPoint)
                    if side:
                        relationship.relationArray.append(( rec.id, side))
                        break
            if len(relationship.relationArray) > 1:
                allRelationships.append(relationship)
        print(json.dumps(allRelationships, cls=Encoder))
        with open("relationship.json", "w") as outfile:
            outfile.write(json.dumps(allRelationships, cls=Encoder, indent="\t"))

        # for list in allRelationships:
        #     for tuple in list.relationArray:
        #         cv2.putText(pr_image,
        #                     tuple[2],
        #                     tuple[0],
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     1, (255, 255, 0), 1)
        scale_percent = 65
        WidndowWidth = int(pr_image.shape[1] * scale_percent / 100)
        height = int(pr_image.shape[0] * scale_percent / 100)

        # dsize
        dsize = (WidndowWidth, height)

        # resize image
        pr_image= cv2.resize(pr_image, dsize)


        cv2.imshow('primage', pr_image)
        #
        # # Exiting the window if 'q' is pressed on the keyboard.
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        return

class Relationship:
    def __init__(self):
        self.relationArray = []
        self.relation = 'Association'

