import cv2
import matplotlib.pyplot as plt
import numpy as np

def getIntersectionPoint(x1, y1, x2, y2, x3, y3, x4, y4):
    a1 = y2-y1
    b1 = x1-x2
    c1 = a1*x1 + b1*y1

    a2 = y4-y3
    b2 = x3-x4
    c2 = a2*x3 + b2*y3

    determinant = a1*b2 - a2*b1

    if determinant == 0:
        return float('inf'), float('inf')
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant

    return x, y

# reading in an image
# image = mpimg.imread('img.jpeg')
images = []
image = cv2.imread('line.png', cv2.IMREAD_COLOR)
image2 = cv2.imread('line4.png', cv2.IMREAD_COLOR)
images.append(image)
images.append(image2)
# image = cv2.resize(image, (960, 540))

for i in range(len(images)):
    image = images[i]
    # grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # canny edge detection
    canny_image = cv2.Canny(gray_image, 100, 200)

    # erosion
    kernel = np.ones((5,5), np.uint8)
    erosion_image = cv2.erode(canny_image, kernel, iterations=1)

    # dilation
    dilation_image = cv2.dilate(canny_image, kernel, iterations=1)

    # printing out some stats and plotting the image
    print('This image is:', type(image), 'with dimensions:', image.shape)

    lines = cv2.HoughLinesP(
        canny_image,
        rho=1,
        theta=np.pi/60,
        threshold=100,
        lines=np.array([]),
        minLineLength=100,
        maxLineGap=30

        # canny_image,
        # rho = 10,
        # theta = np.pi / 60,
        # threshold = 100,
        # lines = np.array([]),
        # minLineLength = 10,
        # maxLineGap = 250
    )

    # plt.figure()

    _, threshold = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)

    # save points in list
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.append([x1, y1])
        points.append([x2, y2])
    print('points')
    print(points)

    # get intersection points
    for line_outer in lines:
        x1, y1, x2, y2 = line_outer[0]
        for line_inner in lines:
            print(line_inner[0])
            x3, y3, x4, y4 = line_inner[0]
            x, y = getIntersectionPoint(x1, y1, x2, y2, x3, y3, x4, y4)
            if x == float('inf') and y == float('inf'):
                print('parallel')
            else:
                for point in points:
                    if abs(point[0] - x) <= 10 and abs(point[1] - y) <= 10:
                        points.remove(point)
                print('deleted points below:')
                print(x, y)

    # delete edge duplicated points (that are really close)
    for point_ in points:
        for point in points:
            if point != point_:
                if abs(point[0] - point_[0]) <= 5 and abs(point[1] - point_[1]) <= 5:
                    points.remove(point)

    print('modified points')
    print(points)

    # draw points
    for point in points:
        image = cv2.circle(image, (point[0], point[1]), radius=5, color=(0, 0, 255), thickness=-1)
    # # show image
    cv2.imshow('image', image)
    #
    # # Exiting the window if 'q' is pressed on the keyboard.
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()