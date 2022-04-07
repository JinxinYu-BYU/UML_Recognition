from PIL.ImageDraw import ImageDraw
from PIL import Image
from PIL import ImageDraw

scale = 2
image_width, image_height = 500, 500
black, white = 0, 255
line_thickness = 1
solid_triangles_path = 'img/solid_triangles/'
solid_diamonds_path = 'img/solid_diamonds/'
hollow_triangles_path = 'img/hollow_triangles/'
hollow_diamonds_path = 'img/hollow_diamonds/'

solid_triangles_dotted_path = 'img/solid_dotted_triangles/'
hollow_triangles_dotted_path = 'img/hollow_dotted_triangles/'
solid_diamonds_dotted_path = 'img/solid_dotted_diamonds/'
hollow_diamonds_dotted_path = 'img/hollow_dotted_diamonds/'


# def draw_diamond():
#     all_images = []
#     fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
#     img_size = 500
#     # img = np.zeros((img_size, img_size), dtype=np.double)
#
#     img = np.full((img_size, img_size), 1.0, dtype=np.double)
#
#     diamond_top_coordinate = [10, 250]
#     diamond_left_coordinate = [200, 490]
#     diamond_right_coordinate = [200, 10]
#     diamond_bottom_coordinate = [390, 250]
#
#     for i in range(100):
#         # fill polygon
#         triangle = np.array((
#             (diamond_top_coordinate[0], diamond_top_coordinate[1]),
#             (diamond_left_coordinate[0], diamond_left_coordinate[1]),
#             (diamond_bottom_coordinate[0], diamond_bottom_coordinate[1]),
#             (diamond_right_coordinate[0], diamond_right_coordinate[1])
#         ))
#         rr, cc = polygon(triangle[:, 0], triangle[:, 1], img.shape)
#         img[rr, cc] = 255
#
#         # fill polygon
#         line_upper_left = [diamond_bottom_coordinate[0] + 100, diamond_top_coordinate[1] - 5]
#         line_upper_right = [diamond_bottom_coordinate[0] + 100, diamond_top_coordinate[1] + 5]
#         line_lower_left = [diamond_bottom_coordinate[1], diamond_top_coordinate[1] - 5]
#         line_lower_right = [diamond_bottom_coordinate[1], diamond_top_coordinate[1] + 5]
#
#         line = np.array((
#             (line_lower_left[0], line_lower_left[1]),
#             (line_lower_right[0], line_lower_right[1]),
#             (line_upper_right[0], line_upper_right[1]),
#             (line_upper_left[0], line_upper_left[1])
#         ))
#
#         rr, cc = polygon(line[:, 0], line[:, 1], img.shape)
#         # rr, cc = rectangle((line[0], line[1]), (line[3], line[2]))
#
#         img[rr, cc] = 255
#
#         ax1.imshow(img, cmap=plt.cm.gray)
#         img = np.reshape(img, (500, 500))
#         data = im.fromarray(img)
#         plt.show()


# def draw_triangle():
#     all_images = []
#     fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(10, 6))
#     img_size = 500
#     img = np.zeros((img_size, img_size), dtype=np.double)
#
#     triangle_top_coordinate = [10, 250]
#     triangle_left_coordinate = [400, 490]
#     triangle_right_coordinate = [400, 10]
#
#     for i in range(100):
#         # fill polygon
#         triangle = np.array((
#             (triangle_top_coordinate[0], triangle_top_coordinate[1]),
#             (triangle_left_coordinate[0], triangle_left_coordinate[1]),
#             (triangle_right_coordinate[0], triangle_right_coordinate[1])
#         ))
#         rr, cc = polygon(triangle[:, 0], triangle[:, 1], img.shape)
#         img[rr, cc] = 1
#
#         # fill polygon
#         line_upper_left = [triangle_left_coordinate[0], triangle_top_coordinate[1] - 5]
#         line_upper_right = [triangle_left_coordinate[0], triangle_top_coordinate[1] + 5]
#         line_lower_left = [triangle_left_coordinate[1], triangle_top_coordinate[1] - 5]
#         line_lower_right = [triangle_left_coordinate[1], triangle_top_coordinate[1] + 5]
#
#         line = np.array((
#             (line_lower_left[0], line_lower_left[1]),
#             (line_lower_right[0], line_lower_right[1]),
#             (line_upper_right[0], line_upper_right[1]),
#             (line_upper_left[0], line_upper_left[1])
#         ))
#         rr, cc = polygon(line[:, 0], line[:, 1], img.shape)
#         # rr, cc = rectangle((line[0], line[1]), (line[3], line[2]))
#
#         img[rr, cc] = 1
#
#         ax1.imshow(img, cmap=plt.cm.gray)
#         img = np.reshape(img, (500, 500))
#         data = im.fromarray(img)
#         # data.convert('rgb').save('test_img/1.png')
#         # Open image and discard alpha channel which makes wheel round rather than square
#         rgb_img = data.convert('RGB')
#
#         # Convert to Lab colourspace
#         # srgb_p = ImageCms.createProfile("sRGB")
#         # lab_p = ImageCms.createProfile("LAB")
#         #
#         # rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
#         # Lab = ImageCms.applyTransform(im, rgb2lab)
#         # Lab.convert('RGB')
#         # Lab.save()
#         # ax1.set_title('No anti-aliasing')
#         # ax1.axis('off')
#         triangle_right_coordinate[0] = triangle_right_coordinate[1] + \
#                                        (img_size - triangle_right_coordinate[0]) \
#                                        / triangle_right_coordinate[0]
#         plt.show()

def draw_solid_triangle():
    top_coordinate = (25, 0)
    left_coordinate = (0, 40)
    right_coordinate = (50, 40)

    line_top_coordinate = (25, 40)
    line_bottom_coordinate = (25, 50)

    filename = 1

    while top_coordinate[1] < (left_coordinate[1] - line_thickness - 20):
        while left_coordinate[0] < (right_coordinate[0] - 20):
            img = Image.new('L', (image_width, image_height), white)
            draw = ImageDraw.Draw(img)
            left_coordinate = (left_coordinate[0] + 3, left_coordinate[1])
            right_coordinate = (right_coordinate[0] - 3, right_coordinate[1])
            draw.polygon([left_coordinate, top_coordinate, right_coordinate], fill='black')  # left-side line
            # draw.line([top_coordinate, right_coordinate], fill=black)  # right-side line
            # draw.line([left_coordinate, right_coordinate], fill=black)  # bottom-side line

            draw.line([line_top_coordinate, line_bottom_coordinate], fill=black, width=line_thickness)  # line

            img.save(solid_triangles_path + str(filename) + '.png')
            filename += 1
            left_coordinate = (left_coordinate[0] + 3, 400)
            right_coordinate = (right_coordinate[0] - 3, 400)

        top_coordinate = (top_coordinate[0], top_coordinate[1] + 3)
        left_coordinate = (0, 400)
        right_coordinate = (500, 400)
        filename += 1


def draw_hollow_triangle():
    top_coordinate = (250, 00)
    left_coordinate = (0, 400)
    right_coordinate = (500, 400)

    line_top_coordinate = (250, 400)
    line_bottom_coordinate = (250, 500)

    filename = 1

    while top_coordinate[1] < (left_coordinate[1] - line_thickness - 30):
        while left_coordinate[0] < (right_coordinate[0] - 20):
            img = Image.new('L', (image_width, image_height), white)
            draw = ImageDraw.Draw(img)
            left_coordinate = (left_coordinate[0] + 3, left_coordinate[1])
            right_coordinate = (right_coordinate[0] - 3, right_coordinate[1])
            draw.line([top_coordinate, left_coordinate], fill=black, width=line_thickness)  # left-side line
            draw.line([top_coordinate, right_coordinate], fill=black, width=line_thickness)  # right-side line
            draw.line([left_coordinate, right_coordinate], fill=black, width=line_thickness)  # bottom-side line

            draw.line([line_top_coordinate, line_bottom_coordinate], fill=black, width=line_thickness)  # line

            img.save(hollow_triangles_path + str(filename) + '.png')
            filename += 1
            left_coordinate = (left_coordinate[0] + 3, 400)
            right_coordinate = (right_coordinate[0] - 3, 400)

        top_coordinate = (top_coordinate[0], top_coordinate[1] + 3)
        left_coordinate = (0, 400)
        right_coordinate = (500, 400)
        filename += 1


def draw_solid_diamond():
    top_coordinate = (250, 0)
    left_coordinate = (0, 200)
    right_coordinate = (500, 200)
    bottom_coordinate = (250, 400)

    line_top_coordinate = (250, 400)
    line_bottom_coordinate = (250, 500)

    filename = 1

    while top_coordinate[1] < (left_coordinate[1] - line_thickness - 10):
        while left_coordinate[0] < (right_coordinate[0] - 50):
            img = Image.new('L', (image_width, image_height), white)
            draw = ImageDraw.Draw(img)

            left_coordinate = (left_coordinate[0] + 3, left_coordinate[1])
            right_coordinate = (right_coordinate[0] - 3, right_coordinate[1])

            draw.polygon([top_coordinate, left_coordinate, bottom_coordinate, right_coordinate],
                         fill=black)  # top-left-side line
            # draw.line([top_coordinate, right_coordinate], fill=black, width=line_thickness)  # top-right-side line
            # draw.line([left_coordinate, bottom_coordinate], fill=black, width=line_thickness)  # bottom-left-side
            # line draw.line([right_coordinate, bottom_coordinate], fill=black, width=line_thickness)  #
            # bottom-right-side line

            draw.line([line_top_coordinate, line_bottom_coordinate], fill=black, width=line_thickness)  # line

            img.save(solid_diamonds_path + str(filename) + '.png')
            filename += 1

        top_coordinate = (top_coordinate[0], top_coordinate[1] + 3)
        left_coordinate = (0, 200)
        right_coordinate = (500, 200)
        filename += 1


def draw_hollow_diamond():
    top_coordinate = (250, 0)
    left_coordinate = (0, 200)
    right_coordinate = (500, 200)
    bottom_coordinate = (250, 400)

    line_top_coordinate = (250, 400)
    line_bottom_coordinate = (250, 500)

    filename = 1

    while top_coordinate[1] < (left_coordinate[1] - line_thickness - 10):
        while left_coordinate[0] < (right_coordinate[0] - 50):
            img = Image.new('L', (image_width, image_height), white)
            draw = ImageDraw.Draw(img)

            left_coordinate = (left_coordinate[0] + 3, left_coordinate[1])
            right_coordinate = (right_coordinate[0] - 3, right_coordinate[1])

            draw.line([top_coordinate, left_coordinate], fill=black, width=line_thickness)  # top-left-side line
            draw.line([top_coordinate, right_coordinate], fill=black, width=line_thickness)  # top-right-side line
            draw.line([left_coordinate, bottom_coordinate], fill=black, width=line_thickness)  # bottom-left-side line
            draw.line([right_coordinate, bottom_coordinate], fill=black, width=line_thickness)  # bottom-right-side line

            draw.line([line_top_coordinate, line_bottom_coordinate], fill=black, width=line_thickness)  # line

            img.save(hollow_diamonds_path + str(filename) + '.png')
            filename += 1

        top_coordinate = (top_coordinate[0], top_coordinate[1] + 3)
        left_coordinate = (0, 200)
        right_coordinate = (500, 200)
        filename += 1


def draw_solid_triangle_dotted():
    top_coordinate = (250, 0)
    left_coordinate = (0, 400)
    right_coordinate = (500, 400)
    line_top_coordinate = (250, 400)

    filename = 1

    while top_coordinate[1] < (left_coordinate[1] - line_thickness - 20):
        while left_coordinate[0] < (right_coordinate[0] - 20):
            img = Image.new('L', (image_width, image_height), white)
            draw = ImageDraw.Draw(img)
            left_coordinate = (left_coordinate[0] + 3, left_coordinate[1])
            right_coordinate = (right_coordinate[0] - 3, right_coordinate[1])
            draw.polygon([left_coordinate, top_coordinate, right_coordinate], fill='black')  # left-side line

            line_length = 15
            line_space = 10
            temp_top_coordinate = (line_top_coordinate[0], line_top_coordinate[1] - line_space)
            temp_end_coordinate = (temp_top_coordinate[0], temp_top_coordinate[1] + line_length)

            while line_length >= 5:
                while temp_end_coordinate[1] < image_height:
                    temp_top_coordinate = (temp_top_coordinate[0], temp_end_coordinate[1] + line_space)
                    temp_end_coordinate = (temp_top_coordinate[0], temp_top_coordinate[1] + line_length)
                    draw.line([temp_top_coordinate, temp_end_coordinate], fill=black, width=line_thickness)  # line

                img.save(solid_triangles_dotted_path + str(filename) + '.png')
                filename += 1

                line_length = line_length - 5
                line_space = line_space - 1

            left_coordinate = (left_coordinate[0] + 3, 400)
            right_coordinate = (right_coordinate[0] - 3, 400)
        top_coordinate = (top_coordinate[0], top_coordinate[1] + 3)
        left_coordinate = (0, 400)
        right_coordinate = (500, 400)
        filename += 1


def draw_solid_diamond_dotted():
    top_coordinate = (250, 0)
    left_coordinate = (0, 200)
    right_coordinate = (500, 200)
    bottom_coordinate = (250, 400)

    line_top_coordinate = (250, 400)
    line_bottom_coordinate = (250, 500)

    filename = 1

    while top_coordinate[1] < (left_coordinate[1] - line_thickness - 10):
        while left_coordinate[0] < (right_coordinate[0] - 50):
            img = Image.new('L', (image_width, image_height), white)
            draw = ImageDraw.Draw(img)

            left_coordinate = (left_coordinate[0] + 3, left_coordinate[1])
            right_coordinate = (right_coordinate[0] - 3, right_coordinate[1])

            draw.polygon([top_coordinate, left_coordinate, bottom_coordinate, right_coordinate],
                         fill=black)  # top-left-side line

            # draw.line([line_top_coordinate, line_bottom_coordinate], fill=black, width=line_thickness)  # line
            line_length = 15
            line_space = 10
            temp_top_coordinate = (bottom_coordinate[0], bottom_coordinate[1] - line_space)
            temp_end_coordinate = (temp_top_coordinate[0], temp_top_coordinate[1] + line_length)

            while line_length >= 5:
                while temp_end_coordinate[1] < image_height:
                    temp_top_coordinate = (temp_top_coordinate[0], temp_end_coordinate[1] + line_space)
                    temp_end_coordinate = (temp_top_coordinate[0], temp_top_coordinate[1] + line_length)
                    draw.line([temp_top_coordinate, temp_end_coordinate], fill=black, width=line_thickness)  # line

                img.save(solid_diamonds_dotted_path + str(filename) + '.png')
                filename += 1

                line_length = line_length - 5
                line_space = line_space - 1

        top_coordinate = (top_coordinate[0], top_coordinate[1] + 3)
        left_coordinate = (0, 200)
        right_coordinate = (500, 200)
        filename += 1


def draw_hollow_triangle_dotted():
    top_coordinate = (250, 0)
    left_coordinate = (0, 400)
    right_coordinate = (500, 400)

    line_top_coordinate = (250, 400)
    line_bottom_coordinate = (250, 500)

    filename = 1

    while top_coordinate[1] < (left_coordinate[1] - line_thickness - 10):
        while left_coordinate[0] < (right_coordinate[0] - 20):
            img = Image.new('L', (image_width, image_height), white)
            draw = ImageDraw.Draw(img)
            left_coordinate = (left_coordinate[0] + 3, left_coordinate[1])
            right_coordinate = (right_coordinate[0] - 3, right_coordinate[1])
            draw.line([top_coordinate, left_coordinate], fill=black, width=line_thickness)  # left-side line
            draw.line([top_coordinate, right_coordinate], fill=black, width=line_thickness)  # right-side line
            draw.line([left_coordinate, right_coordinate], fill=black, width=line_thickness)  # bottom-side line

            line_length = 15
            line_space = 10
            temp_top_coordinate = (line_top_coordinate[0], line_top_coordinate[1] - line_space)
            temp_end_coordinate = (temp_top_coordinate[0], temp_top_coordinate[1] + line_length)

            while line_length >= 5:
                while temp_end_coordinate[1] < image_height:
                    temp_top_coordinate = (temp_top_coordinate[0], temp_end_coordinate[1] + line_space)
                    temp_end_coordinate = (temp_top_coordinate[0], temp_top_coordinate[1] + line_length)
                    draw.line([temp_top_coordinate, temp_end_coordinate], fill=black, width=line_thickness)  # line

                img.save(hollow_triangles_dotted_path + str(filename) + '.png')
                filename += 1

                line_length = line_length - 5
                line_space = line_space - 1

            left_coordinate = (left_coordinate[0] + 3, 400)
            right_coordinate = (right_coordinate[0] - 3, 400)
        top_coordinate = (top_coordinate[0], top_coordinate[1] + 3)
        left_coordinate = (0, 400)
        right_coordinate = (500, 400)
        filename += 1


def draw_hollow_diamond_dotted():
    top_coordinate = (250, 0)
    left_coordinate = (0, 200)
    right_coordinate = (500, 200)
    bottom_coordinate = (250, 400)

    line_top_coordinate = (250, 400)
    line_bottom_coordinate = (250, 500)

    filename = 1

    while top_coordinate[1] < (left_coordinate[1] - line_thickness - 10):
        while left_coordinate[0] < (right_coordinate[0] - 50):
            img = Image.new('L', (image_width, image_height), white)
            draw = ImageDraw.Draw(img)

            left_coordinate = (left_coordinate[0] + 3, left_coordinate[1])
            right_coordinate = (right_coordinate[0] - 3, right_coordinate[1])

            draw.line([top_coordinate, left_coordinate], fill=black, width=line_thickness)  # top-left-side line
            draw.line([top_coordinate, right_coordinate], fill=black, width=line_thickness)  # top-right-side line
            draw.line([left_coordinate, bottom_coordinate], fill=black, width=line_thickness)  # bottom-left-side line
            draw.line([right_coordinate, bottom_coordinate], fill=black, width=line_thickness)  # bottom-right-side line

            # draw.line([line_top_coordinate, line_bottom_coordinate], fill=black, width=line_thickness)  # line
            line_length = 15
            line_space = 10
            temp_top_coordinate = (bottom_coordinate[0], bottom_coordinate[1] - line_space)
            temp_end_coordinate = (temp_top_coordinate[0], temp_top_coordinate[1] + line_length)

            while line_length >= 5:
                while temp_end_coordinate[1] < image_height:
                    temp_top_coordinate = (temp_top_coordinate[0], temp_end_coordinate[1] + line_space)
                    temp_end_coordinate = (temp_top_coordinate[0], temp_top_coordinate[1] + line_length)
                    draw.line([temp_top_coordinate, temp_end_coordinate], fill=black, width=line_thickness)  # line

                img.save(hollow_diamonds_dotted_path + str(filename) + '.png')
                filename += 1

                line_length = line_length - 5
                line_space = line_space - 1

        top_coordinate = (top_coordinate[0], top_coordinate[1] + 3)
        left_coordinate = (0, 200)
        right_coordinate = (500, 200)
        filename += 1


if __name__ == "__main__":
    draw_solid_diamond()
    draw_solid_triangle()
    draw_hollow_diamond()
    draw_hollow_triangle()
    draw_solid_triangle_dotted()
    draw_solid_diamond_dotted()
    draw_hollow_triangle_dotted()
    draw_hollow_diamond_dotted()

