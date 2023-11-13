import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4988619_segment.png')
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4263361_segment.png')
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-5618336_segment.png')
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-38377437_segment.png')

# Apply bilateral filter
bilateral_filtered_image = cv2.bilateralFilter(image, 15, 75, 75)

# Reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 5  # Number of clusters
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers to uint8 and reshape back to the original image shape
centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

# Now apply k-means using the bilaterally filtered image
bilateral_pixel_values = bilateral_filtered_image.reshape((-1, 3))
bilateral_pixel_values = np.float32(bilateral_pixel_values)

# Apply kmeans
_, bilateral_labels, (bilateral_centers) = cv2.kmeans(bilateral_pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convert centers to uint8 and reshape back to the original image shape
bilateral_centers = np.uint8(bilateral_centers)
bilateral_segmented_image = bilateral_centers[bilateral_labels.flatten()]
bilateral_segmented_image = bilateral_segmented_image.reshape(image.shape)

# Show the images
plt.figure(figsize=(16, 8))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
plt.title('K-Means Segmented')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(bilateral_segmented_image, cv2.COLOR_BGR2RGB))
plt.title('Bilateral + K-Means Segmented')

plt.tight_layout()
plt.show()