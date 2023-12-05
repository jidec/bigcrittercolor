import cv2
from bigcrittercolor.helpers.image import _imgToColorPatches,_reconstructImgFromPPD
import numpy as np

img = cv2.imread("D:/bcc/beetles/segments/INATRANDOM-177740239_segment.png")
ppd = _imgToColorPatches(img,return_patch_masks_colors_imgshapes =True)
#recon = _reconstructImgFromPPD(ppd,is_patch_data=True,show=True)

# shape is always at index 2
shape = ppd[2]
img = np.zeros(shape=shape)

patch_bool_arrays = ppd[0]
patch_colors = ppd[1]

bool_array = patch_bool_arrays[0]
color = patch_colors[0]
test = zip(patch_colors,patch_bool_arrays)

# draw the patches
for color, patch_bool_array in zip(patch_colors, patch_bool_arrays):
    img[patch_bool_array] = color
