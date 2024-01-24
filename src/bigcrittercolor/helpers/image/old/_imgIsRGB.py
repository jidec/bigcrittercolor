import numpy as np

def _imgIsRGB(img):
    if len(img.shape) == 3:
        return True