import numpy as np
import cv2
import math

img = cv2.imread('dart_images/dart1.jpg',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
cv2.imwrite( "erosion/dart1.jpg", erosion)
