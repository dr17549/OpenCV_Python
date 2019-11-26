import numpy as np
import cv2
import sys
import math
import os
import collections
from shapely.geometry import LineString
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

np.set_printoptions(threshold=sys.maxsize)

def detect_and_frame(img_output,img_grey):
    cascadePath = "dart_board.xml"

    # normailising light
    cascade = cv2.CascadeClassifier(cascadePath)
    cv2.equalizeHist( img_grey, img_grey )
    faceRect = cascade.detectMultiScale( img_grey, scaleFactor=1.1, minNeighbors=1, minSize=(50,50), maxSize=(500,500),flags=cv2.CASCADE_SCALE_IMAGE )
    color = (255,0,0)
    # for (x,y,width,height) in faceRect:
        # x is row , y is col
        # cv2.rectangle(img_output, (x,y), (x+width,y+height) , color, 2)
    print("VJ : Finished the viola jones detection")
    return faceRect

def convolution(img,kernel):

    image_convoluted = np.zeros((height, width), np.float32)
    kernelX = 1
    kernelY = 1
    img_padded = cv2.copyMakeBorder(img, kernelX, kernelX, kernelY, kernelY, cv2.BORDER_REPLICATE)
    for i in range(1,height):
        for j in range(1,width):
            cov = np.multiply(kernel,img_padded[i - 1 : i + 2, j - 1: j + 2])
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

def sobel(img,dx,dy):

    image_dx = convolution(img,dx)
    image_dy = convolution(img,dy)

    gradient_magnitude = np.sqrt(np.power(image_dx,2) + np.power(image_dx,2))
    gradient_angle = np.arctan2(image_dy,image_dx)

    return image_dx,image_dy,gradient_magnitude,gradient_angle


def circle_detection_hough_space(gradient_magnitude,gradient_angle, hough_space_gradient_threshold, hough_circle_threshold, radius, twod_circle_space):

    final_output = collections.defaultdict(dict)
    hough_space = np.zeros((height, width, radius), np.int32)
    hough_space_output = np.zeros((height, width), np.float32)
    hough_space_max_r = np.zeros((height, width), np.float32)

    print("CD : Calculating Hough Space 3D")
    for i in range(height):
        for j in range(width):
            if(gradient_magnitude[i][j] > hough_space_gradient_threshold):
                for radi in range(radius):
                    x0 = i + int(radi * math.sin(gradient_angle[i][j]))
                    y0 = j + int(radi * math.cos(gradient_angle[i][j]))
                    if x0 >= 0 and y0 >= 0 and x0 < height and y0 < width:
                        hough_space[x0][y0][radi] += 1

                    x1 = i - int(radi * math.sin(gradient_angle[i][j]))
                    y1 = j - int(radi * math.cos(gradient_angle[i][j]))
                    if x1 >= 0 and y1 >= 0 and x1 < height and y1 < width:
                        hough_space[x1][y1][radi] += 1

    print("CD : Calculating 2D Sum of Hough Space and max Radius")
    count = 0
    for i in range(0,height):
        for j in range(0,width):
                hough_space_output[i][j] = np.sum(hough_space[i][j])
                hough_space_max_r[i][j] = np.argmax(hough_space[i][j])
                if hough_space_output[i][j] > hough_circle_threshold:
                    # stored as dictionary
                    final_output[count]['row'] = i
                    final_output[count]['column'] = j
                    count += 1

                    color = (0,0,255)
                    # cv2.circle(twod_circle_space, (j,i), hough_space_max_r[i][j], color, 1)
    # cv2.imwrite("circle_space_" + num + "_.png", twod_circle_space)

    # PLOT GRAPH
    imgplot = plt.imshow(hough_space_output)
    plt.show()
    return final_output, count


def line_detection_hough_space(gradient_magnitude, gradient_angle, hough_line_gradient_threshold, hough_line_threshold):
    thetas = np.deg2rad(np.arange(-90, 90))
    theta_idx = np.nonzero(thetas)
    rho_max = np.ceil(np.sqrt(math.pow(width, 2) + math.pow(height, 2)))
    rhos = np.linspace(0, int(rho_max), int(rho_max))
    hough_space = np.zeros((len(rhos), len(thetas)), np.float32)
    intersection_map = np.zeros((height, width), np.float32)
    lines = []

    print("LD : Calculating rho and theta ")
    for i in range(height):
        for j in range(width):
            if(gradient_magnitude[i][j] > hough_line_gradient_threshold):
                for t in theta_idx[0]:
                    rho = round(i * math.sin(thetas[t]) + j * math.cos(thetas[t]))
                    hough_space[rho][t] += 1

    line_idx = 0
    print("LD : Calculating x1,y1 and x2,y2 ")
    for r in range(int(rho_max)):
        for t in theta_idx[0]:
            if(hough_space[r][t] > hough_line_threshold):
                a = np.cos(thetas[t])
                b = np.sin(thetas[t])
                x0 = a*r
                y0 = b*r
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                lines.append([line_idx, (x1, y1),(x2, y2)])
                line_idx += 1

                cv2.line(twod_circle_space,(x1,y1),(x2,y2),(0,0,255),1)
    # cv2.imwrite("line_space_" + num + "_.png", twod_circle_space)
    print("LD : Calculating intersection ")
    intersection_count = 0
    for line_1 in lines:
        for line_2 in lines:
            if(line_1[0] < line_2[0]):
                line1 = LineString([line_1[1], line_1[2]])
                line2 = LineString([line_2[1], line_2[2]])
                intersect = line1.intersection(line2)
                if not intersect.is_empty and intersect.geom_type == 'Point':
                    if int(intersect.y) < height and int(intersect.x) < width and int(intersect.y) >= 0 and int(intersect.x) >= 0:
                        intersection_count += 1
                        intersection_map[int(intersect.y)][int(intersect.x)] += 1

    # PLOT
    imgplot = plt.imshow(hough_space)
    plt.show()
    return intersection_map, intersection_count

def filter_output(faceRect, circle_dict, circle_iterations, intersection_map, img_output, intersection_count, img_grey):
    # score threshold for 65_threshold_output
    print("FO : Start Filtering")
    line_detected_threshold = intersection_count * 0.01
    circle_detected_threshold = circle_iterations * 0.03
    # decision here
    min_grey = img_grey.min()
    max_grey = img_grey.max()
    grey_counter_threshold = ((max_grey - min_grey) * 0.4) + min_grey

    # iterate each box of viola jones dartboard detection
    for (x,y,width,height) in faceRect:
        center_of_box_row = y + int(height/2)
        center_of_box_col = x + int(width/2)
        circle_count = 0
        line_count = 0
        white_count = 0

        circle_weight = 0.18
        line_weight = 0.7
        grey_weight = 0.12

        # calculate whiteness score
        for i in range(y, y + height):
            for j in range(x, x + width):
                if img_grey[i][j] < grey_counter_threshold:
                    white_count += 1

        # calculate score for lines intersection inside the viola jones box
        for i in range(y, y + height):
            for j in range(x, x + width):
                distance = np.sqrt(np.power((i - center_of_box_row),2) + np.power((j - center_of_box_col),2))
                individual_weight = (100 - distance) / 100
                line_count += int(intersection_map[i][j] * individual_weight)

        # calculate score for circle inside the viola jones box
        for circle_index in range(circle_iterations):
            row = circle_dict[circle_index]['row']
            col = circle_dict[circle_index]['column']

            # track count by giving it a weight of how far it is from the center of the circle
            if row > y and row < y + height and col > x and col < x + width:
                distance = np.sqrt(np.power((row - center_of_box_row),2) + np.power((col - center_of_box_col),2))
                individual_weight = (100 - distance) / 100
                circle_count += int(100 * individual_weight)

        # ignore the circles if there are not inside
        if circle_count == 0:
            c_threshold = 0
        else:
            c_threshold = circle_detected_threshold

        grey_detected_threshold = 0.3 * (height * width)
        total_threshold = (line_weight * line_detected_threshold) + (circle_weight * c_threshold) + (grey_weight * grey_detected_threshold)

        if (line_weight * line_count) + (circle_weight * circle_count) + (grey_weight * white_count) > total_threshold:
            # if more than threshold , draw
            color = (0,255,0)
            cv2.rectangle(img_output, (x,y), (x+width,y+height) , color, 2)

# Main function
if __name__ == "__main__":

    for number in range(8,9):
        num = str(number)
        print(" ------ Calculating :  " + num  + " -----------")
        input = 'dart_images/dart' + num + '.jpg'
        img = cv2.imread(input, 0)
        img_grey = cv2.imread(input, 0)
        img_output = cv2.imread(input, 1)
        height, width = img.shape

        twod_circle_space = np.zeros((height, width,3), dtype=np.uint8)

        dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
        dy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.int32)

        radius = int(width / 2)
        hough_space_gradient_threshold = 85
        hough_circle_threshold = 20
        hough_line_gradient_threshold = 66
        hough_line_threshold = 50


        # viola jones
        # faceRect = detect_and_frame(img,img_grey)
        # calculate all the convolutions pre-calculations
        image_dx,image_dy,gradient_magnitude,gradient_angle = sobel(img,dx,dy)
        # calculation line detection and 65_threshold_output results
        # intersection_map, intersection_count = line_detection_hough_space(gradient_magnitude, gradient_angle, hough_line_gradient_threshold, hough_line_threshold)
        # calculate circle detection
        circle_dict, circle_count = circle_detection_hough_space(gradient_magnitude,gradient_angle, hough_space_gradient_threshold, hough_circle_threshold, radius, twod_circle_space)
        # implement pipeline and filter of detections
        # filter_output(faceRect, circle_dict, circle_count, intersection_map, img_output, intersection_count, img_grey)
        # write 65_threshold_output on to image
        # cv2.imwrite("66_include_grey/output_" + num + "_.png", img_output)



