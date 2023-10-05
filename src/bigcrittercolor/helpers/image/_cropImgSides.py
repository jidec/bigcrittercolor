
# lr_crop_px is the number of pixels left and right of middle_x to SAVE, any past it are cropped
def _cropImgSides(img,middle_x,lr_crop_px):
    left_bound = middle_x - lr_crop_px
    right_bound = middle_x + lr_crop_px
    img = img[:, left_bound:right_bound]
    return img