
def _chopImgLR(img,middle_x,n_chop_px):
    left_bound = middle_x - n_chop_px
    right_bound = middle_x + n_chop_px
    img = img[:, left_bound:right_bound]
    return img