import cv2
import numpy as np

# Converts a 3-channel BGR or RGB image to a 4-channel PNG by making black pixels transparent based on a threshold
def _blackBgToTransparent(img, threshold=0):
    # Make sure the image has an alpha channel
    if img.shape[2] < 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Calculate the sum of all channels
    channel_sum = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]

    # Set the alpha channel to fully transparent for pixels below the threshold
    img[channel_sum <= threshold, 3] = 0

    return img
