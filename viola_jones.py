import numpy as np
import cv2
import sys
import math

img = cv2.imread('dart_images/dart14.jpg', 1)
img_grey = cv2.imread('dart_images/dart14.jpg', 0)


def detect_and_frame(img,img_grey):
    cascadePath = "dart_board.xml"
    # normailising light
    cascade = cv2.CascadeClassifier(cascadePath)
    cv2.equalizeHist( img_grey, img_grey )
    faceRect = cascade.detectMultiScale( img_grey, scaleFactor=1.1, minNeighbors=1, minSize=(50,50), maxSize=(500,500),flags=cv2.CASCADE_SCALE_IMAGE )
    color = (0,255,0)
    for (x,y,width,height) in faceRect:
        cv2.rectangle(img, (x,y), (x+width,y+height) , color, 2)
    return img

detect_and_frame(img,img_grey)
cv2.imwrite( "dart_images/detected.jpg", img)