import cv2
import numpy as np
import argparse
import imutils
import glob

from imutils.object_detection import non_max_suppression


def multi_scale():
    ap = argparse.ArgumentParser()
    templates = ["img/just_triangle.png", "img/just_triangle_left.png", "img/just_triangle_right.png",
                 "img/just_triangle_up.png", "img/just_diamond.png"]
    # ap.add_argument("-t", "--template", required=True, help="help")
    ap.add_argument("-i", "--images", required=True,
                    help="help")
    ap.add_argument("-v", "--visualize",
                    help="Flag indicating whether or not to visualize each iteration")
    ap.add_argument("-b", "--threshold", type=float, default=0.8,
                    help="threshold for multi-template matching")
    args = vars(ap.parse_args())

    threshold = 0

    image = cv2.imread(args["images"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imS = cv2.resize(image, (900, 600))  # Resize image
    cv2.imshow("original", imS)  # Show image
    for current in templates:
        if current == "img/just_triangle.png" or current == "img/just_triangle_left.png" or \
                current == "img/just_triangle_right.png" or current == "img/just_triangle_up.png" or\
                current == "img/single_hollow_arrow.png":
            threshold = 890000
        elif current == "img/just_diamond.png":
            threshold = 2100000

        template = cv2.imread(current)
        # load the image image, convert it to grayscale, and detect edges
        # template = cv2.imread(args["template"])
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]
        
        # loop over the images to find the template in
        # for imagePath in glob.glob(args["images"]):
        # load the image, convert it to grayscale, and initialize the
        # bookkeeping variable to keep track of the matched region

        found = None
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            # resize the image according to the scale, and keep track
            # of the ratio of the resizing
            resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            # if the resized image is smaller than the template, then break
            # from the loop
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # detect edges in the resized, grayscale image and apply template
            # matching to find the template in the image
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            # check to see if the iteration should be visualized
            if args.get("visualize", False):
                # draw a bounding box around the detected region
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                              (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)
            # if we have found a new maximum correlation value, then update
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
                new_result = result
        # unpack the bookkeeping variable and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        # (_, maxLoc, r) = found
        # (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        # (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        # # draw a bounding box around the detected result and display the image
        # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

        # find all locations in the result map where the matched value is
        # greater than the threshold, then clone our original image so we
        # can draw on it
        (yCoords, xCoords) = np.where(new_result >= threshold)
        print("[INFO] {} matched locations *before* NMS".format(len(yCoords)))

        # initialize our list of rectangles
        rects = []

        # loop over the starting (x, y)-coordinates again
        for (x, y) in zip(xCoords, yCoords):
            # update our list of rectangles
            rects.append((x, y, x + tW, y + tH))

        # apply non-maxima suppression to the rectangles
        pick = non_max_suppression(np.array(rects))
        print("[INFO] {} matched locations *after* NMS".format(len(pick)))

        # loop over the final bounding boxes
        for (startX, startY, endX, endY) in pick:
            # draw the bounding box on the image
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (255, 0, 0), 3)

    # cv2.namedWindow("output", cv2.WINDOW_NORMAL) 
    # Create window with freedom of dimensions
    imS = cv2.resize(image, (900, 600))  # Resize image
    cv2.imshow("output", imS)  # Show image
    cv2.waitKey(0)