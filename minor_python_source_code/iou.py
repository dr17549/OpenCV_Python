# This script calculates the IOU of belief detection zone and the actual detection zone
# The format of the script is "0 0 3 3 2 2 4 4 #shape name"
# Run the script with the input txt file as argument such as python iou.py dart4.txt

from collections import namedtuple
import os
import sys
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy
    return 0
  
f_name = sys.argv[1].split('.')[0] 
r_name = f_name + "_result.txt"

with open(sys.argv[1], 'r') as f:
    with open(r_name, 'w+') as o:
        x = f.read().splitlines()
        for i in x:
            xys = i.split('#')[0].split(' ')
            cmt = i.split('#')[1]

            ra = Rectangle(int(xys[0]), int(xys[1]), int(xys[2]), int(xys[3]))
            rb = Rectangle(int(xys[4]), int(xys[5]), int(xys[6]), int(xys[7]))

            belief = abs(ra.ymax - ra.ymin)*abs(ra.xmax - ra.xmin)
            detect = abs(rb.ymax - rb.ymin)*abs(rb.xmax - rb.xmin)
            iou_b = area(ra, rb)/belief*100
            iou_d = area(ra, rb)/detect*100

            o.write(cmt + "\n")
            o.write("IOU over belief: %f\n" % (iou_b))

            if(detect > belief):
                o.write("Detected area is larger \r\n")
            else:
                o.write("Belief area is larger \r\n")