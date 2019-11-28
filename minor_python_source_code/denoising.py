import numpy as np
import cv2
num = '8'
img = cv2.imread('dart_images/dart' + num + '.jpg')
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
cv2.imwrite("de_noised/dart" + num + ".jpg", dst)
