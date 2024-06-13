from bigcrittercolor.helpers.verticalize import _verticalizeImg
import numpy as np
from bigcrittercolor.helpers import _showImages

def _passesHWRatio(rgb_mask, hw_ratio_minmax, show=False):
    mask_vert = _verticalizeImg(rgb_mask, strategy="polygon", show=show)

    _showImages(show, [rgb_mask,mask_vert], titles=["Mask to Filter", "Verticalized Mask to Filter"])
    # if hw_ratio less than min or greater than max, skip
    h = np.shape(mask_vert)[0]
    w = np.shape(mask_vert)[1]
    hw_ratio = h / w

    if hw_ratio < hw_ratio_minmax[0] or hw_ratio > hw_ratio_minmax[1]:
        return False
    else:
        return True
