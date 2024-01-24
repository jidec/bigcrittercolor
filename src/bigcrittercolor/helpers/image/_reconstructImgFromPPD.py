import numpy as np

from bigcrittercolor.helpers import _showImages

def _reconstructImgFromPPD(ppd, input_colorspace="rgb", is_patch_data=True,show=False):

    # shape is always at index 2
    shape = ppd[2]
    img = np.zeros(shape=shape,dtype=np.uint8)

    c1 = 0
    c2 = 0
    c3 = 0

    if input_colorspace == "cielab":
        c1 = 0
        c2 = 128
        c3 = 128

    img[:, :, 0] = c1  # Red channel
    img[:, :, 1] = c2  # Green channel
    img[:, :, 2] = c3  # Blue channel

    if is_patch_data:
        patch_bool_arrays = ppd[0]
        patch_colors = ppd[1]

        # draw the patches
        for color, patch_bool_array in zip(patch_colors, patch_bool_arrays):
            img[patch_bool_array] = color

    else:
        pixel_coords = ppd[0]
        pixel_colors = ppd[1]

        for color, coord in zip(pixel_colors, pixel_coords):
            img[coord[1], coord[0]] = color

    _showImages(show,[img],["Reconstructed"])

    return img