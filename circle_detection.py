import numpy as np
import cv2
import sys
import math
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

np.set_printoptions(threshold=sys.maxsize)

input = 'dart_images/dart14.jpg'
img = cv2.imread(input,0)
img_grey = cv2.imread(input,0)
img_output = cv2.imread(input,1)
height, width = img.shape

dx = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]], np.int32)
dy = np.array([[-1, -2, -1], [0, 0, 0],[1, 2, 1]], np.int32)

radius = int(width/2)

hough_space = np.zeros((height,width,radius),np.int32)
hough_space_output = np.zeros((height,width),np.float32)
hough_space_max_r = np.zeros((height,width),np.float32)
hough_space_gradient_threshold = 100

def detect_and_frame(img,img_grey):
    cascadePath = "dart_board.xml"
    # normailising light
    cascade = cv2.CascadeClassifier(cascadePath)
    cv2.equalizeHist( img_grey, img_grey )
    faceRect = cascade.detectMultiScale( img_grey, scaleFactor=1.1, minNeighbors=1, minSize=(50,50), maxSize=(500,500),flags=cv2.CASCADE_SCALE_IMAGE )
    color = (0,255,0)
    for (x,y,width,height) in faceRect:
        cv2.rectangle(img_output, (x,y), (x+width,y+height) , color, 2)
    return img

def convolution(img,kernel):

    image_convoluted = np.zeros((height, width), np.float32)
    kernelX = 1
    kernelY = 1
    img_padded = cv2.copyMakeBorder(img, kernelX, kernelX, kernelY, kernelY, cv2.BORDER_REPLICATE)
    for i in range(1,height):
        for j in range(1,width):
            cov = np.multiply(kernel,img_padded[i - 1 : i + 2, j - 1: j + 2])
#             cov = 0
#             for dx_1 in range(-1,2,1):
#                 for dx_2 in range(-1,2,1):
#                     cov = cov + (img_padded[i + dx_1][j + dx_2] * kernel[dx_1 + 1][dx_2+1])
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


def circle_detection_hough_space(gradient_magnitude,gradient_angle, hough_space_gradient_threshold):

    hough_space = np.zeros((height, width, radius), np.int32)
    hough_space_output = np.zeros((height, width), np.float32)
    hough_space_max_r = np.zeros((height, width), np.float32)

    print("Start calculating hough space 3D")
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

    print("DONE : Cal 3D Hough Space")

    for i in range(0,height):
        for j in range(0,width):
                hough_space_output[i][j] = np.sum(hough_space[i][j])
                hough_space_max_r[i][j] = np.argmax(hough_space[i][j])
                if hough_space_output[i][j] > 18:
                    color = (255,0,0)
                    cv2.circle(img_output, (j,i), hough_space_max_r[i][j], color, 1)



def line_detection_hough_space(gradient_magnitude, gradient_angle):
    thetas = np.deg2rad(np.arange(-90, 90))
    theta_idx = np.nonzero(thetas)
    rho_max = np.ceil(np.sqrt(math.pow(width, 2) + math.pow(height, 2)))
    rhos = np.linspace(0, rho_max, rho_max)
    hough_space = np.zeros((len(rhos), len(thetas)), np.float32)
    threshold = 65

    for i in range(height):
        for j in range(width):
            if(gradient_magnitude[i][j] > threshold):
                for t in theta_idx[0]:
                    rho = round(i * math.sin(thetas[t]) + j * math.cos(thetas[t]))
                    hough_space[rho][t] += 1
    # print(hough_space)

    for r in range(int(rho_max)):
        for t in theta_idx[0]:
            if(hough_space[r][t] > 38):
                a = np.cos(thetas[t])
                b = np.sin(thetas[t])
                x0 = a*r
                y0 = b*r
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(img_output,(x1,y1),(x2,y2),(255,0,0),1)


# image_dx,image_dy,gradient_magnitude,gradient_angle = sobel(img,dx,dy)
# line_detection_hough_space(gradient_magnitude, gradient_angle)
# # circle_detection_hough_space(gradient_magnitude,gradient_angle, hough_space_gradient_threshold)
# cv2.imwrite('output.png',img_output)

detect_and_frame(img,img_grey)
cv2.imwrite( "output.png", img_output)



