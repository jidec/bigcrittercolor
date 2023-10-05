import numpy as np
import cv2
from bigcrittercolor.helpers import _showImages

def _getBlobLineAvgDarkness(src_img, line_start, line_end, show=False):

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