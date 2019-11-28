import numpy as np
import cv2
import sys
import math
import os
import collections
import sys
from collections import namedtuple
from shapely.geometry import LineString
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

np.set_printoptions(threshold=sys.maxsize)

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    return 0


def rectangle(rectangle1, rectangle2):
    ra = Rectangle(int(rectangle1['xmin']), int(rectangle1['ymin']), int(rectangle1['xmin'] + rectangle1['width']),
                   int(rectangle1['ymin'] + rectangle1['height']))
    rb = Rectangle(int(rectangle2['xmin']), int(rectangle2['ymin']), int(rectangle2['xmin'] + rectangle2['width']),
                   int(rectangle2['ymin'] + rectangle2['height']))

    area_a = abs(ra.ymax - ra.ymin) * abs(ra.xmax - ra.xmin)
    area_b = abs(rb.ymax - rb.ymin) * abs(rb.xmax - rb.xmin)
    iou = area(ra, rb)
    smaller = 'a'

    if iou == 0:
        return 0, smaller

    ref = min(area_a, area_b)
    if (ref == area_a):
        smaller = 'a'
        # a is inside b
        if (iou / area_a == 1):
            return 2, smaller
    if (ref == area_b):
        smaller = 'b'
        # a is inside b
        if (iou / area_b == 1):
            return 3, smaller

    return iou / area_a, smaller


def detect_and_frame(img_output, img_grey):
    cascadePath = "dart_board.xml"

    # normailising light
    cascade = cv2.CascadeClassifier(cascadePath)
    cv2.equalizeHist(img_grey, img_grey)
    faceRect = cascade.detectMultiScale(img_grey, scaleFactor=1.1, minNeighbors=1, minSize=(50, 50), maxSize=(500, 500),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    color = (255, 0, 0)
    # for (x,y,width,height) in faceRect:
    # x is row , y is col
    # cv2.rectangle(img_output, (x,y), (x+width,y+height) , color, 2)
    return faceRect


def convolution(img, kernel):
    image_convoluted = np.zeros((height, width), np.float32)
    kernelX = 1
    kernelY = 1
    img_padded = cv2.copyMakeBorder(img, kernelX, kernelX, kernelY, kernelY, cv2.BORDER_REPLICATE)
    for i in range(1, height):
        for j in range(1, width):
            cov = np.multiply(kernel, img_padded[i - 1: i + 2, j - 1: j + 2])
            image_convoluted[i][j] = float(np.sum(cov) / 9)

    return image_convoluted


def normalize(image_dx):
    min = image_dx.min()
    max = image_dx.max()

    for i in range(0, height):
        for j in range(0, width):
            # if min < 0:
            # image_dx[i][j] = image_dx[i][j] + np.abs(min)
            image_dx[i][j] = (image_dx[i][j] - min) * 255 / (max - min)

    return image_dx


def sobel(img, dx, dy):
    image_dx = convolution(img, dx)
    image_dy = convolution(img, dy)

    gradient_magnitude = np.sqrt(np.power(image_dx, 2) + np.power(image_dx, 2))
    gradient_angle = np.arctan2(image_dy, image_dx)

    return image_dx, image_dy, gradient_magnitude, gradient_angle


def circle_detection_hough_space(gradient_magnitude, gradient_angle, hough_space_gradient_threshold,
                                 hough_circle_threshold, radius, twod_circle_space, height,width,x_start,y_start):
    final_output = collections.defaultdict(dict)
    hough_space = np.zeros((height, width, radius), np.int32)
    hough_space_output = np.zeros((height, width), np.float32)
    hough_space_max_r = np.zeros((height, width), np.float32)

    for i in range(y_start,height):
        for j in range(x_start,width):
            if (gradient_magnitude[i][j] > hough_space_gradient_threshold):
                for radi in range(radius):
                    x0 = i + int(radi * math.sin(gradient_angle[i - y_start][j - x_start]))
                    y0 = j + int(radi * math.cos(gradient_angle[i - y_start][j - x_start]))
                    if x0 >= 0 and y0 >= 0 and x0 < height and y0 < width:
                        hough_space[x0][y0][radi] += 1

                    x1 = i - int(radi * math.sin(gradient_angle[i - y_start][j - x_start]))
                    y1 = j - int(radi * math.cos(gradient_angle[i - y_start][j - x_start]))
                    if x1 >= 0 and y1 >= 0 and x1 < height and y1 < width:
                        hough_space[x1][y1][radi] += 1

    count = 0
    for i in range(0, height):
        for j in range(0, width):
            hough_space_output[i][j] = np.sum(hough_space[i][j])
            hough_space_max_r[i][j] = np.argmax(hough_space[i][j])
            if hough_space_output[i][j] > hough_circle_threshold:
                # stored as dictionary
                final_output[count]['row'] = i
                final_output[count]['column'] = j
                count += 1

                # color = (0, 0, 255)
                # cv2.circle(twod_circle_space, (j,i), hough_space_max_r[i][j], color, 1)
    # cv2.imwrite("circle_space_" + num + "_.png", twod_circle_space)

    # PLOT GRAPH
    # imgplot = plt.imshow(hough_space_output)
    # plt.show()
    return final_output, count


def line_detection_hough_space(gradient_magnitude, gradient_angle, hough_line_gradient_threshold, hough_line_threshold, height, width,x_start,y_start):
    thetas = np.deg2rad(np.arange(-90, 90))
    theta_idx = np.nonzero(thetas)
    rho_max = np.ceil(np.sqrt(math.pow(width, 2) + math.pow(height, 2)))
    rhos = np.linspace(0, int(rho_max), int(rho_max))
    hough_space = np.zeros((len(rhos), len(thetas)), np.float32)
    intersection_map = np.zeros((height, width), np.float32)
    lines = []

    for i in range(y_start,height):
        for j in range(x_start,width):
            if (gradient_magnitude[i - y_start][j - x_start] > hough_line_gradient_threshold):
                for t in theta_idx[0]:
                    rho = round(i * math.sin(thetas[t]) + j * math.cos(thetas[t]))
                    hough_space[rho][t] += 1

    line_idx = 0
    for r in range(int(rho_max)):
        for t in theta_idx[0]:
            if (hough_space[r][t] > hough_line_threshold):
                a = np.cos(thetas[t])
                b = np.sin(thetas[t])
                x0 = a * r
                y0 = b * r
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                lines.append([line_idx, (x1, y1), (x2, y2)])
                line_idx += 1

                cv2.line(twod_circle_space, (x1, y1), (x2, y2), (0, 0, 255), 1)
    # cv2.imwrite("line_space_" + num + "_.png", twod_circle_space)
    intersection_count = 0
    for line_1 in lines:
        for line_2 in lines:
            if (line_1[0] < line_2[0]):
                line1 = LineString([line_1[1], line_1[2]])
                line2 = LineString([line_2[1], line_2[2]])
                intersect = line1.intersection(line2)
                if not intersect.is_empty and intersect.geom_type == 'Point':
                    if int(intersect.y) < height and int(intersect.x) < width and int(intersect.y) >= 0 and int(
                            intersect.x) >= 0:
                        intersection_count += 1
                        intersection_map[int(intersect.y)][int(intersect.x)] += 1

    # PLOT
    # imgplot = plt.imshow(hough_space)
    # plt.show()
    return intersection_map, intersection_count


def filter_output(faceRect, img_output, img_grey):
    # score threshold for 65_threshold_output
    all_positive_areas = collections.defaultdict(dict)
    positive_boxes_count = 0
    print("FO : Start Filtering")
    average_score = 0
    # decision here
    min_grey = img_grey.min()
    max_grey = img_grey.max()
    grey_counter_threshold = ((max_grey - min_grey) * 0.4) + min_grey
    all_scores = []
    # iterate each box of viola jones dartboard detection
    # reduce the number of boxes first
    for (x, y, width, height) in faceRect:
        all_positive_areas[positive_boxes_count]['xmin'] = x
        all_positive_areas[positive_boxes_count]['ymin'] = y
        all_positive_areas[positive_boxes_count]['width'] = width
        all_positive_areas[positive_boxes_count]['height'] = height
        all_positive_areas[positive_boxes_count]['skip'] = False
        positive_boxes_count += 1

    # take out the areas that intersect too much
    for iterative in range(positive_boxes_count):
        for inner_iteration in range(positive_boxes_count):
            if iterative != inner_iteration:
                if not all_positive_areas[inner_iteration]['skip'] and not all_positive_areas[iterative]['skip']:
                    area_intersect, smaller = rectangle(all_positive_areas[inner_iteration],
                                                        all_positive_areas[iterative])
                    if area_intersect > 0.5:
                        if smaller == 'a':
                            smaller_index = inner_iteration
                            larger_index = iterative

                        else:
                            smaller_index = iterative
                            larger_index = inner_iteration
                        # make other iteration skip this one
                        all_positive_areas[smaller_index]['skip'] = True

                        all_positive_areas[larger_index]['xmin'] = min(all_positive_areas[iterative]['xmin']
                                                                       , all_positive_areas[inner_iteration]['xmin'])
                        all_positive_areas[larger_index]['ymin'] = min(all_positive_areas[iterative]['ymin'],
                                                                       all_positive_areas[inner_iteration]['ymin'])
                        all_positive_areas[larger_index]['width'] = max(all_positive_areas[iterative]['width'],
                                                                        all_positive_areas[inner_iteration]['width'])
                        all_positive_areas[larger_index]['height'] = max(all_positive_areas[iterative]['height'],
                                                                         all_positive_areas[inner_iteration]['height'])


    for iterative in range(positive_boxes_count):
        y = all_positive_areas[iterative]['ymin']
        x = all_positive_areas[iterative]['xmin']
        width = all_positive_areas[iterative]['width']
        height = all_positive_areas[iterative]['height']
        center_of_box_row = y + int(height / 2)
        center_of_box_col = x + int(width / 2)

        circle_count = 0
        line_count = 0
        white_count = 0

        circle_weight = 0.35
        line_weight = 0.60
        grey_weight = 0.05

        circle_dict, circle_iterations = circle_detection_hough_space(gradient_magnitude, gradient_angle,
                                                                      hough_space_gradient_threshold,
                                                                      hough_circle_threshold,
                                                                      radius, twod_circle_space,height,width,y,x)

        intersection_map, intersection_count = line_detection_hough_space(gradient_magnitude, gradient_angle,
                                                                          hough_line_gradient_threshold,
                                                                          hough_line_threshold,height,width,y,x)
        # calculate whiteness score
        for i in range(y, y + height):
            for j in range(x, x + width):
                if img_grey[i][j] < grey_counter_threshold:
                    white_count += 1

        individual_acc_score = (line_weight * intersection_count) + (circle_weight * circle_iterations) + (grey_weight * white_count)
        all_positive_areas[iterative]['score'] = individual_acc_score
        all_scores = all_scores + [individual_acc_score]
        average_score += individual_acc_score

    # calculate average score
    average_score = int(average_score / positive_boxes_count) - (max(all_scores) - min(all_scores))

    for iterative in range(positive_boxes_count):
        if not all_positive_areas[iterative]['skip'] and all_positive_areas[iterative]['score'] >= average_score:
            color = (0, 255, 0)
            cv2.rectangle(img_output, (all_positive_areas[iterative]['xmin'], all_positive_areas[iterative]['ymin']),
                          (all_positive_areas[iterative]['xmin'] + all_positive_areas[iterative]['width'],
                           all_positive_areas[iterative]['ymin'] + all_positive_areas[iterative]['height']), color, 2)


# Main function
if __name__ == "__main__":

    for number in range(0, 16):
        num = str(number)
        print(" ------ Calculating :  " + num + " -----------")
        # for denoising images
        # input_name = 'dart_images/dart' + num + '.jpg'
        # input = cv2.imread(input_name)
        # img = cv2.fastNlMeansDenoisingColored(input, None, 10, 10, 7, 21)
        # img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img_output = cv2.imread(input_name, 1)
        # height, width, third_d = input.shape

        # normal detection
        input = 'de_noised/dart' + num + '.jpg'
        img = cv2.imread(input, 0)
        img_grey = cv2.imread(input, 0)
        img_output = cv2.imread(input, 1)
        height, width = img.shape

        twod_circle_space = np.zeros((height, width, 3), dtype=np.uint8)

        dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
        dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.int32)

        radius = int(width / 2)
        hough_space_gradient_threshold = 85
        hough_circle_threshold = 20
        hough_line_gradient_threshold = 66
        hough_line_threshold = 50

        # viola jones
        faceRect = detect_and_frame(img, img_grey)

        image_dx, image_dy, gradient_magnitude, gradient_angle = sobel(img, dx, dy)

        # implement pipeline and filter of detections
        filter_output(faceRect, img_output, img_grey)
        # write 65_threshold_output on to image
        cv2.imwrite("new_algo/" + num + "_.png", img_output)



