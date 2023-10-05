import cv2
import numpy as np
from bigcrittercolor.helpers import _showImages

# finds the longest line that can be drawn through the sole white blob in a binary image by:
# 1. getting the minimum area rectangle
# 2. finding all points in the top face, AND the bot face
# 3. drawing lines between every top face point and every bottom face point
# 4. finding the line containing the most white pixels (the longest line)

# returns a tuple of points for the line
# GREYSCALE uint8 image is taken
def _getLongestLineThruBlob2(binary_img, epsilon_mult=0.02, show=False):

    # find contour (polygon) from binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # keep largest just in case there are multiple
    largest_contour = max(contours, key=cv2.contourArea)
    # reduce complexity of polygon
    epsilon = epsilon_mult * cv2.arcLength(largest_contour, True)  # 2% of contour perimeter
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

    if show:
        poly_img = np.zeros_like(binary_img)
        cv2.drawContours(poly_img, [polygon], -1, (255), 2)

        _showImages(show,poly_img,["Image Simplified to Polygon"])

    # fix format
    polygon = [item[0] if len(item) == 1 else item for item in polygon]
    polygon = np.array(polygon, dtype=np.int32)

    line = _lineAcrossPolygon(polygon)


    if show:
        cv2.line(poly_img, line[0], line[1], color=(255, 0, 0), thickness=3)
        _showImages(show,poly_img,["Polygon with Line"])

    return line

def _lineAcrossPolygon(polygon):
    def is_inside_polygon(polygon, point):
        return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0

    def point_percent_between(p1, p2, percent):
        t = percent / 100.0
        x1, y1 = p1
        x2, y2 = p2

        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2

        return (x, y)

    # Use brute force approach: For each vertex of the polygon,
    # compute the distance to every other vertex and keep track of the maximum.

    max_distance = 0
    start = None
    end = None
    for i in range(len(polygon)):
        for j in range(len(polygon)):
            if i != j:
                distance = cv2.norm(np.array(polygon[i]) - np.array(polygon[j]))
                if distance > max_distance and is_inside_polygon(polygon,point_percent_between(polygon[i],polygon[j],25)) and is_inside_polygon(polygon,point_percent_between(polygon[i],polygon[j],50)) and is_inside_polygon(polygon,point_percent_between(polygon[i],polygon[j],75)): #and is_inside_polygon(polygon, (polygon[i] + polygon[j]) / 2.0):
                    start = polygon[i]
                    end = polygon[j]
                    max_distance = distance

    return [start, end]
#binary_img = cv2.imread('D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/masks/INAT-215236-2_mask.png', cv2.IMREAD_GRAYSCALE)
#_getLongestLineThruBlob2(binary_img,show=True)