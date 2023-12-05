import cv2
from bigcrittercolor.helpers import _showImages

def _blur(img, type, ksize=5, sigma_x=0, d=9, sigma_color=75, sigma_space=75, auto_adjust_blur=False, show=False):
    """
    Apply a blur to an image

    :param img: The input image.
    :param type: Type of blur ('gaussian', 'median', 'bilateral').
    :param ksize: Kernel size for Gaussian and Median blur. Defaults to 5.
    :param sigma_x: Gaussian kernel standard deviation in X direction.
    :param d: Diameter of each pixel neighborhood for Bilateral filter.
    :param sigma_color: Filter sigma in the color space for Bilateral filter.
    :param sigma_space: Filter sigma in the coordinate space for Bilateral filter.
    :return: Blurred image.
    """

    # Adjust blur parameters based on image size
    if auto_adjust_blur:
        height, width = img.shape[:2]
        scale_factor = max(height, width) / 1000  # Scale factor based on larger dimension
        ksize = max(3, int(ksize * scale_factor))  # Ensure ksize is at least 3
        sigma_x = sigma_x * scale_factor if sigma_x != 0 else 0
        d = max(3, int(d * scale_factor))  # Ensure d is at least 3
        sigma_color = sigma_color * scale_factor
        sigma_space = sigma_space * scale_factor

    if type == 'gaussian':
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma_x)
    elif type == 'median':
        blurred = cv2.medianBlur(img, ksize)
    elif type == 'bilateral':
        blurred = cv2.bilateralFilter(img, d, sigma_color, sigma_space)

    _showImages(show, [img, blurred], ['Image', 'Blurred'])
    
    return blurred