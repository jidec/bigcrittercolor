import cv2
from bigcrittercolor.helpers.verticalize import _isLineInsidePoly
from bigcrittercolor.helpers import _showImages
import math
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
import numpy as np

# Strategies:
# polygon - simplify blob to a polygon, lines are then all lines between polygon points that don't pass outside the poly
# ellipse - fastest, fit an ellipse to the blob and return only the major axis line of that ellipse, note that metric ranking becomes trivial bc there is only one line
# skeleton_hough - skeletonize the blob then get straight-ish lines using a hough lines transform, number of lines is dependent on resolution
#   often produces less trivial lines than polygon and faster because of this

def _getLinesAcrossBlob(greyu8_img, strategy="polygon", poly_epsilon_mult=0.01,
                        sh_rho=1,sh_theta=np.pi/30,sh_thresh=25,show=False):
    match strategy:
        # polygon simplifies the blob to a poly then gets all lines between polygon points
        #   that lie within the blob
        case "polygon":
            contours, _ = cv2.findContours(greyu8_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # keep largest just in case there are multiple
            largest_contour = max(contours, key=cv2.contourArea)
            # reduce complexity of polygon using epsilon
            epsilon = poly_epsilon_mult * cv2.arcLength(largest_contour, True)  # 2% of contour perimeter
            polygon = cv2.approxPolyDP(largest_contour, epsilon, True)

            lines = []
            for i in range(len(polygon)):
               for j in range(len(polygon)):
                   pt1 = (polygon[i][0][0], polygon[i][0][1])
                   pt2 = (polygon[j][0][0], polygon[j][0][1])
                   line = (pt1,pt2)
                   #print(line)
                   if _isLineInsidePoly(line, polygon):
                       lines.append(line)
            return lines

        # ellipse fits an ellipse around the blob then takes the major axis line
        case "ellipse":
            ret, thresh = cv2.threshold(greyu8_img, 3, 255, cv2.THRESH_BINARY)

            # find contours
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return None
            
            # get the biggest contour and fit an ellipse to it
            big_contour = max(contours, key=cv2.contourArea)

            if len(big_contour) < 5:
                return None

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
            return [line]

        case "skeleton_hough":
            skeleton = skeletonize(greyu8_img/255)
            skeleton = img_as_ubyte(skeleton)

            lines = cv2.HoughLines(skeleton, 1, np.pi / 30, 25)  # Lower the threshold #25

            img_fixed1 = np.copy(greyu8_img)
            img_fixed2 = np.copy(greyu8_img)

            lines_fixed1 = [] # fixed 1 contains lines reformatted from houghlines to start/end pts
            lines_fixed2 = [] # fixed 2 contains these lines starting at the start and end of the blob
            # visualize HoughLines
            if lines is not None:
                for line in lines:

                    # reformat lines for lines_fixed1
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    lines_fixed1.append(((x1,y1),(x2,y2)))
                    if show:
                        cv2.line(img_fixed1, (x1, y1), (x2, y2), (255, 255, 255), 2)

                    # fix for fix2
                    line_img = np.zeros_like(greyu8_img)
                    cv2.line(line_img, (x1,y1), (x2,y2), color=255, thickness=3)
                    masked = cv2.bitwise_and(greyu8_img, greyu8_img, mask=line_img.astype(np.uint8)) # mask line img over image

                    white_pixel_coords = np.argwhere(masked == 255) # get coords of white pixels
                    if np.shape(white_pixel_coords)[0] == 0:
                        continue
                    topmost_pt = tuple(white_pixel_coords[0][::-1]) # extract the topmost and bottommost white pixels
                    bottommost_pt = tuple(white_pixel_coords[-1][::-1])

                    lines_fixed2.append((topmost_pt,bottommost_pt))
                    if show:
                        cv2.line(img_fixed2, topmost_pt, bottommost_pt, color=0, thickness=3)

                if show:
                    _showImages(show,[greyu8_img,skeleton,img_fixed1,img_fixed2],["Mask","Skeleton","Hough Lines","Lines Fixed"])

            return lines_fixed2
