import numpy as np
import cv2

from bigcrittercolor.helpers.verticalize import _verticalizeImg

def _blobPassesFilter(rgb_mask, hw_ratio_minmax=None, prop_img_minmax=None, rot_invar_sym_min=None, intersects_sides=False, prevert=False, show=False):
    """ Given a binary rgb mask containing one blob, return True if it passes the provided filters

        Args:
            rgb_mask (array): binary rgb mask containing one blob
            hw_ratio_minmax (tuple):
            prop_img_minmax (tuple):
            rot_invar_sym_min (double):
    """
    if hw_ratio_minmax is not None:
        mask = np.copy(rgb_mask)
        if not prevert:
            mask = _verticalizeImg(mask, strategy="polygon", show=show)
        # if hw_ratio less than min or greater than max, skip
        h = np.shape(mask)[0]
        w = np.shape(mask)[1]
        hw_ratio = h / w

        if hw_ratio < hw_ratio_minmax[0] or hw_ratio > hw_ratio_minmax[1]:
            return False

    if prop_img_minmax is not None:
        white_pixel_count = cv2.countNonZero(
            cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY))  # get number of white pixels as nonzeros

        total_area = rgb_mask.shape[0] * rgb_mask.shape[1]  # get total number of pixels as height * width

        white_area_percent = (white_pixel_count / total_area)  # get percent of image covered by white

        if white_area_percent < prop_img_minmax[0] or white_area_percent > prop_img_minmax[1]:
            return False

    if rot_invar_sym_min is not None:
        greyu8 = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        #sym = _getRotInvariantBlobSymmetry(greyu8, show=show)
        #if sym < sym_min:
        #    return False

    if intersects_sides:
        grey_img = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY)
        # Check top and bottom borders
        if np.any(grey_img[0, :] == 255) or np.any(grey_img[-1, :] == 255):
            return False

        # Check left and right borders
        if np.any(grey_img[:, 0] == 255) or np.any(grey_img[:, -1] == 255):
            return False

    return True
