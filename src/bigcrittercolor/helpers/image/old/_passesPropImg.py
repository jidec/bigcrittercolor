import cv2

def _passesPropImg(rgb_mask, prop_minmax):
    white_pixel_count = cv2.countNonZero(cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2GRAY))  # get number of white pixels as nonzeros

    total_area = rgb_mask.shape[0] * rgb_mask.shape[1]  # get total number of pixels as height * width

    white_area_percent = (white_pixel_count / total_area)  # get percent of image covered by white

    if white_area_percent < prop_minmax[0] or white_area_percent > prop_minmax[1]:
        return False
    else:
        return True
