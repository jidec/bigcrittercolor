import cv2
import numpy as np
from bigcrittercolor.helpers import _showImages
from bigcrittercolor.helpers.verticalize import _vertUsingLine

# new rot invariant blob sym function
def _getRotInvariantBlobSymmetry(binary_img, epsilon_mult=0.02, return_sym_img=False, show=False):
    def is_inside_polygon(polygon, point):
        return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0

    def point_percent_between(p1, p2, percent):
        t = percent / 100.0
        x1 = p1[0][0]
        y1 = p1[0][1]
        x2 = p2[0][0]
        y2 = p2[0][1]

        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2

        return (x, y)

    # find contour (polygon) from binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # keep largest just in case there are multiple
    largest_contour = max(contours, key=cv2.contourArea)
    # reduce complexity of polygon
    epsilon = epsilon_mult * cv2.arcLength(largest_contour, True)  # 2% of contour perimeter
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Use brute force approach: For each vertex combo of the polygon,
    # compute the symmetry and return the line of best symmetry

    #print("Started looking at each vertex combo")
    #print(str(len(polygon)))
    max_sym = 0
    start = None
    end = None
    for i in range(len(polygon)):
        for j in range(len(polygon)):
            if i != j:
                sym_score = _getSymByLineSidesPixelCount(binary_img, polygon[i],polygon[j])
                if sym_score > max_sym and is_inside_polygon(polygon,point_percent_between(polygon[i],polygon[j],25)) and is_inside_polygon(polygon,point_percent_between(polygon[i],polygon[j],50)) and is_inside_polygon(polygon,point_percent_between(polygon[i],polygon[j],75)): #and is_inside_polygon(polygon, (polygon[i] + polygon[j]) / 2.0):
                    start = (polygon[i][0][0],polygon[i][0][1])
                    end = (polygon[j][0][0],polygon[j][0][1])
                    max_sym = sym_score

    #print("Done looking at each vertex combo")
    if return_sym_img:
        img = cv2.merge([binary_img,binary_img,binary_img])
        return(_vertUsingLine(img,line=[start,end],show=show))

    return max_sym

# get a sym score calculated as the number of white pixels on either side of a line across a binary image
def _getSymByLineSidesPixelCount(binary_img, line_start, line_end):
    image = binary_img

    # Get the shape of the image
    height, width = image.shape

    # Line equation: y = mx + c
    x0 = line_start[0][0]
    y0 = line_start[0][1]
    x1 = line_end[0][0]
    y1 = line_end[0][1]

    m = (y1 - y0) / (x1 - x0 + 1e-10)  # Slope of the line
    c = y0 - m * x0  # y-intercept

    # Compute the x and y grids
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute the position array using the formula for the cross product
    position = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)

    # Find the white pixels in the image
    white_pixels = (image == 255)

    # Count white pixels on each side
    left_count = np.sum(np.logical_and(white_pixels, position > 0))
    right_count = np.sum(np.logical_and(white_pixels, position < 0))

    max_count = max([left_count,right_count])
    min_count = min([left_count,right_count])

    sym_score = min_count / max_count
    return sym_score