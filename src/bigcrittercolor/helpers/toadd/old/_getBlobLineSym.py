import cv2
import numpy as np
from bigcrittercolor.helpers.imgtransforms import _vertUsingLine
from bigcrittercolor.helpers.imgtransforms import _narrowToBoundingRect
from bigcrittercolor.helpers import _showImages

def _getBlobLineSym(binary_img, line_start,line_end, use_count=False, show=False):
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