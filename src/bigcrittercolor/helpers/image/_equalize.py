import cv2
import numpy as np
from bigcrittercolor.helpers import _showImages

def _equalize(img, input_colorspace="rgb", type='clahe', clip_limit=2.0, tile_grid_size=(8, 8), show=False):
    """
    Apply equalization techniques to an image.

    :param img: The input image.
    :param type: Equalization method ('histogram' or 'clahe').
    :param clip_limit: Clip limit for CLAHE. Defaults to 2.0.
    :param tile_grid_size: Tile grid size for CLAHE. Defaults to (8, 8).
    :return: Equalized image.
    """

    if type == 'histogram':
        if len(img.shape) == 2:
            # Grayscale image
            equalized_img = cv2.equalizeHist(img)
        else:
            # Color image
            equalized_img = np.copy(img)
            for i in range(3):  # Apply histogram equalization on each channel
                equalized_img[:, :, i] = cv2.equalizeHist(img[:, :, i])

    elif type == 'clahe':
        # Convert RGB to CIELAB if necessary
        if input_colorspace == 'rgb':
            img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        else:
            img_lab = img.copy()

        # Split the channels
        l, a, b = cv2.split(img_lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_eq = clahe.apply(l)

        # Merge channels back
        img_lab_eq = cv2.merge((l_eq, a, b))

        # Convert back to RGB if the original image was RGB
        if input_colorspace == 'rgb':
            equalized_img = cv2.cvtColor(img_lab_eq, cv2.COLOR_LAB2RGB)
        else:
            equalized_img = img_lab_eq

    _showImages(show, [img, equalized_img], ['Image', 'Equalized'])

    return equalized_img