import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
#image = cv2.imread('D:/bcc/ringtails/masks/INAT-17015184-1_mask.png')
image = cv2.imread('D:/bcc/ringtails/masks/INAT-17036129-1_mask.png')
image = cv2.imread('D:/bcc/ringtails/masks/INAT-50722953-1_mask.png')
image = cv2.imread('D:/bcc/ringtails/masks/INAT-51463867-1_mask.png')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

def local_contrast(image, window_size=15):
    mean = cv2.blur(image, (window_size, window_size))
    sqr_mean = cv2.blur(image**2, (window_size, window_size))
    variance = sqr_mean - mean**2
    contrast = np.sqrt(variance)
    return contrast

def gradient_magnitude(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)
    return grad_magnitude

def compute_glcm_homogeneity(image, distances=[5], angles=[0]):
    glcm = greycomatrix(image, distances, angles, 256, symmetric=True, normed=True)
    homogeneity = greycoprops(glcm, 'homogeneity')
    return homogeneity.mean()

# Load the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Local Contrast
local_contrast_map = local_contrast(gray_image)
plt.figure()
plt.title("Local Contrast")
plt.imshow(local_contrast_map, cmap='gray')
plt.colorbar()
plt.show()

# Gradient Magnitude
grad_magnitude = gradient_magnitude(gray_image)
plt.figure()
plt.title("Gradient Magnitude")
plt.imshow(grad_magnitude, cmap='gray')
plt.colorbar()
plt.show()

# GLCM Homogeneity
homogeneity = compute_glcm_homogeneity(gray_image)
print(f"GLCM Homogeneity: {homogeneity}")