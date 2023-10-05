import numpy as np
import cv2
from bigcrittercolor.helpers.image import _narrowToBoundingRect
from bigcrittercolor.helpers.verticalize import _vertUsingLine
from bigcrittercolor.helpers import _showImages

def _get_blob_line_overlap_sym(binary_img,line,show=False):
    binary_3_channel = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

    vert_img_and_line = _vertUsingLine(binary_3_channel, line, return_img_and_line=True, show=False)
    vert_img = vert_img_and_line[0]
    cX = vert_img_and_line[1][0][0]
    #print(vert_img_and_line[0])
    #print(vert_img_and_line[1])
    if cX == 0:
        return 0
    left_side = _narrowToBoundingRect(vert_img[:, :cX])
    right_side = _narrowToBoundingRect(np.fliplr(vert_img[:, cX:]))
    h, w = left_side.shape[:2]
    right_side = cv2.resize(right_side, (w, h))
    overlap = np.sum(np.bitwise_and(left_side, right_side))

    symmetry_score = overlap / (np.sum(left_side) + np.sum(right_side))
    symmetry_score = round(symmetry_score, 2)

    _showImages._showImages(show, [left_side, right_side, vert_img], ["Left","Right",str(symmetry_score) + " sym"])

    return symmetry_score

def _get_blob_line_npix_sym(binary_img, line, show=False):
    binary_3_channel = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

    vert_img_and_line = _vertUsingLine(binary_3_channel, line, return_img_and_line=True, show=False)
    vert_img = vert_img_and_line[0]
    cX = vert_img_and_line[1][0][0]

    left_side = _narrowToBoundingRect(vert_img[:, :cX])
    right_side = _narrowToBoundingRect(np.fliplr(vert_img[:, cX:]))
    h, w = left_side.shape[:2]
    right_side = cv2.resize(right_side, (w, h))

    white_l = np.sum(left_side == 255)
    white_r = np.sum(right_side == 255)
    score1 = white_r / white_l
    score2 = white_l / white_r
    symmetry_score = min([score1, score2])
    symmetry_score = round(symmetry_score, 2)

    _showImages._showImages(show, [left_side, right_side, vert_img], ["Left","Right",str(symmetry_score) + " sym"])

    return symmetry_score

def _get_blob_line_avg_darkness(src_img, line):
    line_start = line[0]
    line_end = line[1]
    line_img = np.zeros_like(src_img)
    cv2.line(line_img,line_start,line_end,color=255,thickness=3)

    masked = cv2.bitwise_and(src_img, src_img, mask=cv2.cvtColor(line_img,cv2.COLOR_RGB2GRAY).astype(np.uint8))
    non_black_mask = np.any(masked != 0, axis=-1)
    non_black_pixels = masked[non_black_mask]
    #print(non_black_pixels)
    mean_color = np.mean(non_black_pixels, axis=0)[0]
    mean_color = round(mean_color,2)

    score = 255 / mean_color

    return score

def _get_blob_line_skinniness(binary_img, line,show=False):

    line_start = line[0]
    line_end = line[1]

    line_img = np.zeros_like(binary_img)
    num_blob_pix = np.sum(binary_img == 255)
    cv2.line(line_img, line_start, line_end, color=255, thickness=int(num_blob_pix / 1000))

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the contour with the largest area (assuming it's the main blob)
    c = max(contours, key=cv2.contourArea)
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    #box = np.int0(box)
    # Extract the ROI from the another image using bounding rectangle for simplicity
    x, y, w, h = cv2.boundingRect(c)
    line_img = line_img[y:y + h, x:x + w]
    binary_img = binary_img[y:y + h, x:x + w]

    masked = cv2.bitwise_and(binary_img, binary_img, mask=line_img.astype(
        np.uint8))  # mask=cv2.cvtColor(line_img,cv2.COLOR_RGB2GRAY).astype(np.uint8))
    num_white_pix_line = np.sum(line_img == 255)
    num_white_pix_masked = np.sum(masked == 255)

    # starts at 1 and probably maxes at 1.5
    skinny = num_white_pix_line / num_white_pix_masked

    _showImages._showImages(show, [binary_img, line_img, masked], ["Image", "Line", str(skinny) + " Masked"])

    return skinny

# Strategies:
#   overlap_sym - the extent to which shapes on either sides of the line overlap in their pixels
#   npix_sym - symmetry but simply based on the number of pixels on either side
#   avg_darkness
#   skinniness

def _getBlobLineMetric(binary_img, line, metric="overlap_sym", src_img=None, show=False):
    match metric:
        # overlap sym looks at how much pixels across the line overlap
        case "overlap_sym":
            score = _get_blob_line_overlap_sym(binary_img, line, show=show)

        # npix sym looks at simply how many white pixels are in each half
        case "npix_sym":
            score = _get_blob_line_npix_sym(binary_img, line, show=show)

        # avg darkness looks at how dark pixels in the line are
        case "avg_darkness":
            score = _get_blob_line_avg_darkness(src_img, line)

        # avg darkness looks at how dark pixels in the line are
        case "skinniness":
            score = _get_blob_line_skinniness(binary_img, line, show=show)

        case "mean_all":
            overlap_sym_score = _get_blob_line_overlap_sym(binary_img, line, show=show)
            npix_sym_score = _get_blob_line_npix_sym(binary_img, line, show=show)
            avg_darkness = _get_blob_line_avg_darkness(src_img, line)
            skinniness = _get_blob_line_skinniness(binary_img, line, show=show)

            lst = [overlap_sym_score,npix_sym_score,avg_darkness,skinniness]

            score = sum(lst) / len(lst)

    return score
