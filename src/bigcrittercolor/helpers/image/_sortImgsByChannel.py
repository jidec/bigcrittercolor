import numpy as np
import cv2
from PIL import Image, ImageOps
import math
from bigcrittercolor.helpers.image import _resizeImgsToAvgSize

def _sortImgsByChannel(images, colorspace="rgb", channel=0, inverse=False):
    """
    Sorts a list of images by the average value of a specified channel in a given color space.

    Args:
        images (list): List of OpenCV images (NumPy arrays).
        colorspace (str): Color space to use for sorting ('rgb', 'cielab', 'hls').
        channel (int): Channel index within the color space to sort by (0-based).
        inverse (bool): Whether to sort in descending (True) or ascending (False) order.

    Returns:
        list: Sorted list of images.
    """
    def average_channel(image, colorspace, channel):
        # Convert to the specified color space
        if colorspace == "rgb":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif colorspace == "cielab":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif colorspace == "hls":
            converted = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        else:
            raise ValueError(f"Unsupported colorspace: {colorspace}")

        # Mask out black (transparent) areas
        mask = (converted[:, :, 0] != 0) | (converted[:, :, 1] != 0) | (converted[:, :, 2] != 0)
        pixels = converted[mask]

        if len(pixels) == 0:  # Handle empty images
            return 0

        # Calculate the average value of the specified channel
        average_value = np.mean(pixels[:, channel])
        return average_value

    # Calculate average channel value for each image
    channel_values = [(image, average_channel(image, colorspace, channel)) for image in images]

    # Sort images based on the channel value
    sorted_images = [img[0] for img in sorted(channel_values, key=lambda x: x[1], reverse=inverse)]

    return sorted_images
