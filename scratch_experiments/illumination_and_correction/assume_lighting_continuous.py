import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from bigcrittercolor.helpers.image import _segToMask

def quantify_shadowing(image, sigma=15):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filter to extract low-frequency component
    low_freq = gaussian_filter(gray, sigma=sigma)

    # get mask
    mask = _segToMask(image)

    cv2.imshow('0',mask)
    cv2.waitKey(0)

    masked_image = cv2.bitwise_and(low_freq, low_freq, mask=mask)
    cv2.imshow('1', masked_image)
    cv2.waitKey(0)

    # Get the height and width of the image
    height, width = mask.shape

    # Create an empty list to store the pixels in the mask
    pixels_in_mask = []

    # Iterate over each pixel in the mask
    for y in range(height):
        for x in range(width):
            if mask[y, x] != 0:
                pixels_in_mask.append(low_freq[y, x])

    # Convert the list to a NumPy array
    pixels_in_mask = np.asarray(pixels_in_mask)

    # Subtract low-frequency component to get high-frequency component
    #high_freq = gray - low_freq

    # Calculate the variance of the low-frequency component
    shadow_variance = np.var(pixels_in_mask)

    return shadow_variance, low_freq


# Load image
image = cv2.imread('D:/bcc/ringtails/segments/INAT-147315-1_segment.png')
#image = cv2.imread('D:/bcc/ringtails/segments/INAT-144691-1_segment.png')

# Quantify shadowing
shadow_variance, low_freq = quantify_shadowing(image)

# Display results
print(f'Shadow Variance: {shadow_variance}')
cv2.imshow('Original Image', image)
cv2.imshow('Low-Frequency Component', low_freq)
cv2.waitKey(0)
cv2.destroyAllWindows()