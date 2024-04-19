import numpy as np

from bigcrittercolor.helpers import _showImages

def _reconstructImgFromPPD(ppd, input_colorspace="rgb", is_patch_data=True,show=False):

    # shape is always at index 2
    shape = ppd[2]

    img = np.zeros(shape=(shape[0],shape[1],3),dtype=np.uint8) # used to just be shape # note that shape is contered from

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
        #print("Shape")
        #print(np.shape(img))

        for color, coord in zip(pixel_colors, pixel_coords):
            #print(coord)
            img[coord[0], coord[1]] = color #swap 0 and 1

    _showImages(show,[img],["Reconstructed"])

    return img