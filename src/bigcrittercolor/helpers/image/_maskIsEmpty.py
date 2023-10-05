import cv2
import numpy as np

def _maskIsEmpty(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    if not np.any(img > 100):
        return True