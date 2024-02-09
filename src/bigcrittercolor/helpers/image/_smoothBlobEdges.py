import numpy as np
import cv2

# input must be a 3 channel binary or color blob
def _smoothBlobEdges(img, base_kernel_size=5, scaling_factor=500):

    is_color = not np.all(np.logical_or(np.all(img == 0, axis=-1), np.all(img == 255, axis=-1)))

    # Non-black pixels are those where at least one channel is > 0
    non_black_pixels = np.sum(np.any(img > 0, axis=-1))
    total_pixels = img.shape[0] * img.shape[1]
    fraction_non_black = non_black_pixels / total_pixels

    # Dynamically adjust the kernel size based on the fraction of non-black pixels
    kernel_size = int(base_kernel_size + scaling_factor * fraction_non_black)

    # Ensure kernel size is odd
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size

    # Apply Gaussian Blur for smoothing
    blurred_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # For binary imgs: apply threshold to maintain binary nature
    # For RGB blobs: this step may be adjusted or skipped based on the application
    if img.max() == 255 and np.unique(img).size <= 2:  # Simple binary check
        _, smoothed_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)
    else:
        smoothed_img = blurred_img  # For RGB blobs, keep the blurred img

    return smoothed_img