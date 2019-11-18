import numpy as np
import cv2
import sys
import math

np.set_printoptions(threshold=sys.maxsize)

# read grey image
img = cv2.imread('coins1.png',0)
img_output = cv2.imread('coins1.png',1 )
height, width = img.shape

dx = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]], np.int32)
dy = np.array([[-1, -2, -1], [0, 0, 0],[1, 2, 1]], np.int32)

radius = int(width/2)

hough_space = np.zeros((height,width,radius),np.int32)
hough_space_output = np.zeros((height,width),np.float32)
hough_space_max_r = np.zeros((height,width),np.float32)
hough_space_gradient_threshold = 90

# Cal DX
# ---------------------------------------------------
def convolution(img,kernel):

    image_convoluted = np.zeros((height, width), np.float32)
    kernelRadiusX = 1
    kernelRadiusY = 1
    img_padded = cv2.copyMakeBorder(img, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY, cv2.BORDER_REPLICATE)
    for i in range(0,height):
        for j in range(0,width):
            cov = 0
            for dx_1 in range(-1,2,1):
                for dx_2 in range(-1,2,1):
                    cov = cov + (img_padded[i + dx_1][j + dx_2] * kernel[dx_1 + 1][dx_2+1])
            image_convoluted[i][j] = float(cov / 9)

    return image_convoluted

# Normalizer dor DX
# ---------------------------------------------------
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

# image_dx = convolution(img,dx)
# image_dy = convolution(img,dy)
# image_dx = normalize(image_dx)
# image_dy = normalize(image_dy)


image_dx,image_dy,gradient_magnitude,gradient_angle = sobel(img,dx,dy)
# gradient_magnitude = normalize(gradient_magnitude)
# gradient_angle = normalize(gradient_angle)


# Calculate the Hough Space Circle Detection
# ---------------------------------------------------
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

for i in range(0,height):
    for j in range(0,width):
        for radi in range(radius):
            hough_space_output[i][j] = hough_space[i][j][radi] + hough_space_output[i][j]

for i in range(0,height):
    for j in range(0,width):
        max = 0
        max_radi = 1
        for radi in range(radius):
            if hough_space[i][j][radi] > max:
                max = hough_space[i][j][radi]
                max_radi = radi
        hough_space_max_r[i][j] = max_radi

for i in range(0,height):
    for j in range(0,width):
        if hough_space_output[i][j] > 15:
            color = (255,0,0)
            cv2.circle(img_output, (j,i), hough_space_max_r[i][j], color, 1)
# ---------------------------------------------------



# print(image_dx)
cv2.imwrite('output.png',img_output)
# cv2.imwrite('output_mag.png',gradient_magnitude)
# cv2.imwrite('output_angle.png',gradient_angle)

# print(gradient_magnitude)


