import cv2

def _is_inside_polygon(polygon, point):
    return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0

def _point_percent_between(p1, p2, percent):
    t = percent / 100.0
    #x1 = p1[0][0]
    #y1 = p1[0][1]
    #x2 = p2[0][0]
    #y2 = p2[0][1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    x = (1 - t) * x1 + t * x2
    y = (1 - t) * y1 + t * y2

    return (x, y)

# return true if a line is INSIDE a polygon, that is points along the line at an interval stay within the polygon
def _isLineInsidePoly(line, polygon):
    if _is_inside_polygon(polygon, _point_percent_between(line[0], line[1], 25)) and \
            _is_inside_polygon(polygon, _point_percent_between(line[0], line[1], 50)) and \
            _is_inside_polygon(polygon, _point_percent_between(line[0], line[1], 75)):
        return True