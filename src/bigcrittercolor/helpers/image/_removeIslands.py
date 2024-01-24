import cv2
import numpy as np

# remove contours other than the largest contour
def _removeIslands(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # if no contour found, skip
    if not contours:
        return img

    else:
        # empty image and fill with big contour
        mask = np.zeros_like(img)
        cv2.drawContours(mask, [max(contours, key=len)], -1, 255, thickness=-1)

        return mask