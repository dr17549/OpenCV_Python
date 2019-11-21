import numpy as np
import cv2

img = cv2.imread('dart_images/dart6.jpg')
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
cv2.imwrite("dart_images/de_noised.jpg", dst)
