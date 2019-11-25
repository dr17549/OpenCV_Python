import cv2


num = '1'
img = cv2.imread('eq_his/dart' + num +'.jpg')
# de noised
dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
cv2.imwrite( "e_and_d/dart" + num + ".jpg", dst)

