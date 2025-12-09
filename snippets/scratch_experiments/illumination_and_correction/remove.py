import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def remove_illumination_variation(image, sigma=10):
    # Apply Gaussian filter to each channel to extract low-frequency components
    low_freq_r = gaussian_filter(image[:, :, 2], sigma=sigma)
    low_freq_g = gaussian_filter(image[:, :, 1], sigma=sigma)
    low_freq_b = gaussian_filter(image[:, :, 0], sigma=sigma)

    # Subtract low-frequency component from original image to get high-frequency component
    high_freq_r = image[:, :, 2] - low_freq_r
    high_freq_g = image[:, :, 1] - low_freq_g
    high_freq_b = image[:, :, 0] - low_freq_b

    # Combine the high-frequency components back into an RGB image
    high_freq_image = cv2.merge([high_freq_b, high_freq_g, high_freq_r])

    # Clip values to range [0, 255] to avoid invalid pixel values
    high_freq_image = np.clip(high_freq_image, 0, 255).astype(np.uint8)

    return high_freq_image, low_freq_r, low_freq_g, low_freq_b


# Load image
image = cv2.imread('D:/bcc/ringtails/segments/INAT-144691-1_segment.png')

# Remove illumination variation
high_freq_image, low_freq_r, low_freq_g, low_freq_b = remove_illumination_variation(image)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('High-Frequency Image', high_freq_image)
cv2.waitKey(0)
cv2.destroyAllWindows()