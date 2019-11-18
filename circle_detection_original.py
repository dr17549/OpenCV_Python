import numpy as np
import cv2
import sys
import math

np.set_printoptions(threshold=sys.maxsize)

# read grey image
img = cv2.imread('coins1.png',0)
img_output = cv2.imread('coins1.png', 1)
height, width = img.shape

dx = np.array([[-1, 0, 1], [-2, 0, 2],[-1, 0, 1]], np.int32)
dy = np.array([[-1, -2, -1], [0, 0, 0],[1, 2, 1]], np.int32)

radius = int(width/2)

image_dx = np.zeros((height,width),np.float32)
gradient_magnitude = np.zeros((height,width),np.float32)
image_dy = np.zeros((height,width),np.float32)
gradient_angle = np.zeros((height,width),np.float32)
hough_space = np.zeros((height,width,radius),np.int32)

hough_space_output = np.zeros((height,width),np.float32)
hough_space_max_r = np.zeros((height,width),np.float32)
hough_space_gradient_threshold = 100


# Cal DX
# ---------------------------------------------------

for i in range(1,height-1):
    for j in range(1,width-1):
        cov = 0
        for dx_1 in range(-1,2,1):
            for dx_2 in range(-1,2,1):
                cov = cov + (img[i + dx_1][j + dx_2] * dx[dx_1 + 1][dx_2+1])
        image_dx[i][j] = float(cov / 9)

# Normalizer dor DX
# ---------------------------------------------------
min = 100
for i in range(1,height-1):
    for j in range(1,width-1):
        if image_dx[i][j] < min:
            min = image_dx[i][j]
print(min)

if min < 0:
    min = 0 - min
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            image_dx[i][j] = image_dx[i][j] + min

# Cal Dy
# ---------------------------------------------------

for i in range(1,height-1):
    for j in range(1,width-1):
        cov = 0
        for dx_1 in range(-1,2,1):
            for dx_2 in range(-1,2,1):
                cov = cov + (img[i + dx_1][j + dx_2] * dy[dx_1 + 1][dx_2+1])
        image_dy[i][j] = float(cov/9)

# Normalizer for DY
# ---------------------------------------------------
# min = 100
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         if image_dy[i][j] < min:
#             min = image_dy[i][j]
#
# if min < 0:
#     print(min)
#     min = 0 - min
#     print(min)
#     for i in range(1, height - 1):
#         for j in range(1, width - 1):
#             image_dy[i][j] = image_dy[i][j] + min
# ---------------------------------------------------

# # Cal Magnitude
# # ---------------------------------------------------
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         dx2 = pow(image_dx[i][j],2)
#         dy2 = pow(image_dy[i][j],2)
#         res = math.sqrt(dx2 + dy2)
#         gradient_magnitude[i][j] = res
#
# # # Cal Angle
# # # ---------------------------------------------------
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         val = np.arctan2(image_dy[i][j],image_dx[i][j])
#         gradient_angle[i][j] = val
# # # ---------------------------------------------------
#
# # Calculate Hough Space
# # -----------------------------
#
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         if(gradient_magnitude[i][j] > hough_space_gradient_threshold):
#
#             for radi in range(radius):
#                 x0 = i + int(radi * math.sin(gradient_angle[i][j]))
#                 y0 = j + int(radi * math.cos(gradient_angle[i][j]))
#                 if x0 >= 0 and y0 >= 0 and x0 < height and y0 < width:
#                     hough_space[x0][y0][radi] = hough_space[x0][y0][radi] + 1
#
#                 x1 = i - int(radi * math.sin(gradient_angle[i][j]))
#                 y1 = j - int(radi * math.cos(gradient_angle[i][j]))
#                 if x1 >= 0 and y1 >= 0 and x1 < height and y1 < width:
#                     hough_space[x1][y1][radi] = hough_space[x1][y1][radi] + 1
#
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         for radi in range(radius):
#             hough_space_output[i][j] = hough_space[i][j][radi] + hough_space_output[i][j]
#
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         max = 0
#         max_radi = 1
#         for radi in range(radius):
#             if hough_space[i][j][radi] > max:
#                 max = hough_space[i][j][radi]
#                 max_radi = radi
#         hough_space_max_r[i][j] = max_radi
#
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         if hough_space_output[i][j] > 5:
#             color = (255,0,0)
#             cv2.circle(img_output, (j,i), hough_space_max_r[i][j], color, 1)

# Normalizer for output
# ---------------------------------------------------
# max = 0
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         if gradient_magnitude[i][j] > max:
#             max = gradient_magnitude[i][j]
#
# min = 100
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         if gradient_magnitude[i][j] < min:
#             min = gradient_magnitude[i][j]
#
#
# for i in range(1, height - 1):
#     for j in range(1, width - 1):
#         gradient_magnitude[i][j] = (gradient_magnitude[i][j] - min) * 255 / (max-min)
# ---------------------------------------------------
# Normalizer for output
# # ---------------------------------------------------
# max = 0
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         if gradient_angle[i][j] > max:
#             max = gradient_angle[i][j]
#
# min = 100
# for i in range(1,height-1):
#     for j in range(1,width-1):
#         if gradient_angle[i][j] < min:
#             min = gradient_angle[i][j]
#
# #
# for i in range(1, height - 1):
#     for j in range(1, width - 1):
#         gradient_angle[i][j] = (gradient_angle[i][j] - min) * 255 / (max-min)
# # ---------------------------------------------------



# print(image_dx)
cv2.imwrite('output_dx.png',img_output)
# cv2.imwrite('output_dx.png',image_dx)
# cv2.imwrite('output_dy.png',image_dy)
# cv2.imwrite('output_mag.png',gradient_magnitude)
# cv2.imwrite('output_ang.png',gradient_angle)

# print(gradient_magnitude)
