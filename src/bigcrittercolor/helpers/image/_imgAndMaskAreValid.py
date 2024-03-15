from bigcrittercolor.helpers.image import _imgIsValid
import numpy as np

def _imgAndMaskAreValid(img, mask):
    if not _imgIsValid(img):
        return False
    if not _imgIsValid(mask):
        return False

    # Check if the mask is of type np.uint8
    if mask.dtype != np.uint8:
        return False

    # Check if the mask has the same dimensions as the source images
    if img.shape[:2] != mask.shape[:2]:
        print("Error: Mask and images do not have the same dimensions.")
        return False

    return True