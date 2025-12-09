import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read the image
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4988619_segment.png')
#image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4263361_segment.png')
#image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-5618336_segment.png')
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-38377437_segment.png')

# Convert to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image, (7, 7), 0)

# Canny Edge Detection
edges_canny = cv2.Canny(blurred, 100, 200)

# Sobel Edge Detection
sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
sobel = sobelx + sobely

# Laplacian of Gaussian
gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Morphological Gradient
kernel = np.ones((5, 5), np.uint8)
gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

# Color-based Segmentation (k-means)
# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# Convert to float
pixel_values = np.float32(pixel_values)
# Define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K = 3  # Number of clusters
_, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Convert back to 8 bit values
centers = np.uint8(centers)
# Map labels to center values
segmented_image = centers[labels.flatten()]
# Reshape back to the original image
segmented_image = segmented_image.reshape(image.shape)

# Visualize the results
plt.figure(figsize=(20, 10))
plt.subplot(231), plt.imshow(image_rgb), plt.title('Original Image')
plt.subplot(232), plt.imshow(edges_canny, cmap='gray'), plt.title('Canny Edges')
plt.subplot(233), plt.imshow(sobel, cmap='gray'), plt.title('Sobel Edges')
plt.subplot(234), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian of Gaussian')
plt.subplot(235), plt.imshow(gradient, cmap='gray'), plt.title('Morphological Gradient')
plt.subplot(236), plt.imshow(segmented_image), plt.title('K-Means Segmentation')

plt.show()