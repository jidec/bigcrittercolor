import cv2
import math

# line of the form ((xtop,ytop),(xbot,ybot)), a tuple of tuples
def _getBlobEllipseLine(img):
    ret, thresh = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)

    # find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # get the biggest contour and fit an ellipse to it
    big_contour = max(contours, key=cv2.contourArea)

    big_ellipse = cv2.fitEllipse(big_contour)

    # get params from ellipse
    (xc, yc), (d1, d2), angle = big_ellipse

    # compute major radius
    rmajor = max(d1, d2) / 2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90
    xtop = xc + math.cos(math.radians(angle)) * rmajor
    ytop = yc + math.sin(math.radians(angle)) * rmajor
    xbot = xc + math.cos(math.radians(angle + 180)) * rmajor
    ybot = yc + math.sin(math.radians(angle + 180)) * rmajor
    line = ((xtop, ytop), (xbot, ybot))

    return line