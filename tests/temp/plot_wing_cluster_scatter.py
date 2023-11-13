from bigcrittercolor.helpers import plotPixels
import cv2

img = cv2.imread("D:/wing-color/data/segments/UA-008126_fore_segment.png")
plotPixels(img,img_colorspace="rgb")