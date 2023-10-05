import cv2
import numpy as np
from bigcrittercolor.helpers import _showImages
from skimage import img_as_ubyte
from skimage.morphology import skeletonize

# This is a hybrid approach that leverages both longest poly line and symmetry
# 1. simplify the image to a polygon
# 2. get the 5 longest lines between polygon points that pass through the blob
# 3. verticalize using the one that yields the highest symmetry

def _normalizeDragonflyBlobs(binary_img, src_img = None, epsilon_mult=0.01, show=False):
    def is_inside_polygon(polygon, point):
        return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0
    def point_percent_between(p1, p2, percent):
        t = percent / 100.0
        x1 = p1[0][0]
        y1 = p1[0][1]
        x2 = p2[0][0]
        y2 = p2[0][1]
        #x1 = p1[0]
        #y1 = p1[1]
        #x2 = p2[0]
        #y2 = p2[1]

        x = (1 - t) * x1 + t * x2
        y = (1 - t) * y1 + t * y2

        return (x, y)

    def distance(pt1, pt2):
        return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

    def point_line_from_houghline(line):
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        return((x1,y1),(x2,y2))

    def line_inside_polygon(line,polygon):
        if is_inside_polygon(polygon, point_percent_between(line[0], line[1], 25)) and \
            is_inside_polygon(polygon, point_percent_between(line[0], line[1], 50)) and \
            is_inside_polygon(polygon, point_percent_between(line[0], line[1], 75)):
                return True

    # find contour (polygon) from binary image
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # keep largest just in case there are multiple
    largest_contour = max(contours, key=cv2.contourArea)
    # reduce complexity of polygon using epsilon
    epsilon = epsilon_mult * cv2.arcLength(largest_contour, True)  # 2% of contour perimeter
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)


    skeleton = img_as_ubyte(skeletonize(img/255))
    lines = cv2.HoughLines(skeleton, 1, np.pi / 30, 25)  # Lower the threshold
    lines = [point_line_from_houghline(line) for line in lines]
    lines = [line for line in lines if line_inside_polygon(line,polygon)]

    # calculate distances between every pair of points
    #distances = []
    #for i in range(len(polygon)):
    #    for j in range(i + 1, len(polygon)):
    #        # calc distance
    #        d = distance(polygon[i][0], polygon[j][0])
    #        # if is inside polygon, keep the dist
    #        if is_inside_polygon(polygon, point_percent_between(polygon[i], polygon[j], 25)) and \
    #                is_inside_polygon(polygon, point_percent_between(polygon[i], polygon[j], 50)) and \
    #                is_inside_polygon(polygon, point_percent_between(polygon[i], polygon[j], 75)):  # and is_inside_polygon(polygon, (polygon[i] + polygon[j]) / 2.0):
    #            distances.append((d, (polygon[i][0], polygon[j][0])))

    # sort distances in descending order
    #distances.sort(key=lambda x: x[0], reverse=True)

    # get the top 3 longest distances and corresponding points
    #top_lines = distances[:20]

    if show:
        longest_lines_img = np.zeros_like(binary_img)

        for (pt1, pt2) in lines:
            cv2.line(longest_lines_img, tuple(pt1), tuple(pt2), (255), 2)  # Green color

        poly_img = np.zeros_like(binary_img)
        cv2.drawContours(poly_img, [polygon], -1, (255), 2)
        _showImages(show,[poly_img,longest_lines_img],["Image Simplified to Polygon","Longest Lines"])

    #return(poly_img)

    best_score = -255
    for (pt1,pt2) in lines:
        #sym_score_ol = _getSymAcrossLine(binary_img, pt1, pt2, show=True) # can try swapping this sym fun
        #print("OL:" + str(sym_score_ol))
        #sym_score_count = _getSymAcrossLine(binary_img, pt1, pt2, use_count=True, show=True) # can try swapping this sym fun
        #print("Count:" + str(sym_score_count))
        # starts at 1 and probably maxes at like 6
        #dark_score = _getLineAvgDarkness(src_img, line_start=pt1, line_end=pt2,show=show)
        #print("Dark:" + str(dark_score))
        # starts at 1 and probably maxes at 1.5
        skinny_score = _getLineSkinniness(binary_img,line_start=pt1,line_end=pt2,show=show)
        print("Skinny:" + str(skinny_score))

        if skinny_score > best_score:
            best_score = skinny_score
            best_line = (pt1,pt2)

    if show:
        final_line_img = np.copy(binary_img)
        cv2.line(final_line_img, best_line[0],best_line[1], color=(0))
        _showImages(show, [final_line_img], ["Final Line Image"])

    binary_3_channel = cv2.cvtColor(binary_img,cv2.COLOR_GRAY2BGR)
    # return start image verticalized using best line we just found
    return _vertUsingLine(binary_3_channel,line=best_line)

from bigcrittercolor.helpers.verticalize import _vertUsingLine
from bigcrittercolor.helpers.image import _narrowToBoundingRect

def _getSymAcrossLine(binary_img, line_start,line_end, use_count=False, show=False):
    binary_3_channel = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

    vert_img_and_line = _vertUsingLine(binary_3_channel,(line_start,line_end),return_img_and_line=True,show=show)
    vert_img = vert_img_and_line[0]
    cX = vert_img_and_line[1][0][0]

    left_side = _narrowToBoundingRect(vert_img[:, :cX])
    right_side = _narrowToBoundingRect(np.fliplr(vert_img[:, cX:]))
    h, w = left_side.shape[:2]
    right_side = cv2.resize(right_side, (w,h))

    if use_count:
        white_l = np.sum(left_side == 255)
        white_r = np.sum(right_side == 255)
        score1 = white_r / white_l
        score2 = white_l / white_r
        symmetry_score = min([score1,score2])
        symmetry_score = round(symmetry_score, 2)
    else:
        # Measure overlap in the "folded" parts
        #min_width = min(left_side.shape[1], right_side.shape[1])
        overlap = np.sum(np.bitwise_and(left_side, right_side))

        symmetry_score = overlap / (np.sum(left_side) + np.sum(right_side))
        symmetry_score = round(symmetry_score,2)

    _showImages(show, [left_side, right_side, vert_img], ["Left","Right",str(symmetry_score) + " sym"])
    return(symmetry_score)

def _getLineAvgDarkness(src_img, line_start, line_end, show=False):

    line_img = np.zeros_like(src_img)
    cv2.line(line_img,line_start,line_end,color=255,thickness=3)

    masked = cv2.bitwise_and(src_img, src_img, mask=cv2.cvtColor(line_img,cv2.COLOR_RGB2GRAY).astype(np.uint8))
    non_black_mask = np.any(masked != 0, axis=-1)
    non_black_pixels = masked[non_black_mask]
    print(non_black_pixels)
    mean_color = np.mean(non_black_pixels, axis=0)[0]
    mean_color = round(mean_color,2)
    _showImages(show,[src_img,line_img,masked],["Image","Line", str(mean_color) + " Color Masked"])

    score = 255 / mean_color

    return score

# the number of black pixels in the axis
def _getLineSkinniness(binary_img, line_start, line_end, show=False):

    line_img = np.zeros_like(binary_img)
    num_blob_pix = np.sum(binary_img == 255)
    print(num_blob_pix)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the contour with the largest area (assuming it's the main blob)
    c = max(contours, key=cv2.contourArea)
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # Extract the ROI from the another image using bounding rectangle for simplicity
    x, y, w, h = cv2.boundingRect(c)
    line_img = line_img[y:y + h, x:x + w]
    binary_img = binary_img[y:y + h, x:x + w]

    cv2.line(line_img,line_start,line_end,color=255,thickness=int(num_blob_pix / 1000))

    masked = cv2.bitwise_and(binary_img, binary_img, mask=line_img.astype(np.uint8)) #mask=cv2.cvtColor(line_img,cv2.COLOR_RGB2GRAY).astype(np.uint8))
    num_white_pix_line = np.sum(line_img == 255)
    num_white_pix_masked = np.sum(masked == 255)

    # starts at 1 and probably maxes at 1.5
    skinny = num_white_pix_line / num_white_pix_masked


    _showImages(show,[binary_img,line_img,masked],["Image","Line", str(skinny) + " Masked"])

    return(skinny)

from bigcrittercolor.helpers import _getIDsInFolder

ids = _getIDsInFolder("D:/bcc/dfly_appr_expr/appr1/masks")
ids = ids[23:]
for id in ids:
    print(id)
    loc = "D:/bcc/dfly_appr_expr/appr1/masks/" + id + "_mask.png"
    img = cv2.imread(loc, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    src_img = cv2.imread("D:/bcc/dfly_appr_expr/appr1/all_images/" + id + ".jpg")
    src_img = cv2.resize(src_img, (src_img.shape[1] // 2, src_img.shape[0] // 2))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Dilate the image to expand the white parts
    img = cv2.dilate(img, kernel, iterations=3)

    cv2.imshow("0",src_img)
    cv2.waitKey(0)
    print("Loaded img")
    img = _normalizeDragonflyBlobs(binary_img=img,src_img=src_img,show=True)
    print("ran")
    cv2.imshow("0",img)
    cv2.waitKey(0)