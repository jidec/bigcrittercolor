from bigcrittercolor.helpers.verticalize import _getLinesAcrossBlob,_getBlobLineMetric, _vertUsingLine
from bigcrittercolor.helpers.image import _narrowToBoundingRect,_flipHeavyToTop,_cropImgSides
from bigcrittercolor.helpers import _showImages
import numpy as np
import cv2

# Note that blob is a deprecated term, when we say blob we now mean segment

# img MUST have 3 channels

# verticalize consists of 4 steps:
# 1. Find lines across the blob that are candidates for serving as the verticalization axis using _getLinesAcrossBlob(strategy=lines_strategy)
# 2. Get scores for each line using _getBlobLineMetric(metric=best_line_metric)
# 3. Rotate the blob to be vertical using the candidate line with the best score using _vertUsingLine()
# 4. Perform final processing like cropping to a rect around the blob and flipping the heaviest side to the top
def _verticalizeImg(img, lines_strategy="skeleton_hough", best_line_metric="overlap_sym",polygon_e_mult=0.01,
                     sh_rho=1,sh_theta=np.pi/30,sh_thresh=25,
                     bound=True, flip=True,
                     src_img=None, input_line=None,
                     return_line=False, return_img_bb_flip=False, show=False):

    start_img = np.copy(img)

    if input_line is None:
        greyu8 = np.copy(img)
        greyu8[np.any(greyu8 > 1, axis=-1)] = [255, 255, 255]
        greyu8 = cv2.cvtColor(greyu8, cv2.COLOR_BGR2GRAY).astype(np.uint8)  # convert to greyscale uint8

        #cv2.imshow('0',greyu8)
        #cv2.waitKey(0)
        #img_for_lines = np.copy(greyu8)
        #img_for_lines[np.any(img_for_lines > 0, axis=-1)] = 255
        #greyu8[np.any(greyu8 > 20, axis=-1)] = 255

        lines = _getLinesAcrossBlob(greyu8_img=greyu8,strategy=lines_strategy,sh_rho=sh_rho,sh_theta=sh_theta,sh_thresh=sh_thresh,show=show)
        if lines is None:
            return img
        line_scores = [_getBlobLineMetric(greyu8,src_img=src_img,line=line,metric=best_line_metric,show=show) for line in lines]
        if len(line_scores) == 0:
            print("No line scores, skipping")
            return img

        best_index = line_scores.index(max(line_scores))
        best_line = lines[best_index]

    else:
        best_line = input_line

    if return_line:
        return best_line

    img_vert = _vertUsingLine(img,best_line,show=show)
    img = np.copy(img_vert)

    if bound:
        bounded,box = _narrowToBoundingRect(img,return_img_and_bb=True)
        img = np.copy(bounded)

    if flip:
        flipped, has_flipped = _flipHeavyToTop(img, return_img_flip=True)
        img = np.copy(flipped)
        if bound:
            _showImages(show, [start_img, img_vert, bounded, flipped],
                    titles=["Start", "Verticalized", "Bounded", "Flipped"])
        else:
            _showImages(show, [start_img, img_vert, flipped],
                        titles=["Start", "Verticalized", "Flipped"])

    if return_img_bb_flip:
        return(img, box, has_flipped)
    return img