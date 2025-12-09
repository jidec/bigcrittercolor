import numpy as np
import cv2
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt


# Load the image
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-5618336_segment.png')
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4988619_segment.png')
image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4263361_segment.png')

# Check if the image was loaded correctly by visualizing it
cv2.imshow('Original Image', image)
cv2.waitKey(0)

# Convert image to RGB (OpenCV uses BGR by default)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Exclude black pixels
mask = np.all(pixel_values == [0, 0, 0], axis=1)
cleaned_pixel_values = pixel_values[~mask]

# Determine the optimal number of clusters
# (You may need to adjust the range based on your specific image)
sse = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cleaned_pixel_values)
    sse.append(kmeans.inertia_)

kl = KneeLocator(range(1, 10), sse, curve='convex', direction='decreasing')
optimal_k = kl.elbow

# Handle the case when optimal_k is None
if optimal_k is None:
    optimal_k = 3  # Set a default value or try a different range

# Apply KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(cleaned_pixel_values)
predicted_labels = kmeans.predict(cleaned_pixel_values)

# Reconstruct the segmented image
segmented_image = np.copy(pixel_values)
segmented_image[~mask] = kmeans.cluster_centers_[predicted_labels].astype(np.uint8)

# Convert the segmented image from 2D back to the original shape
segmented_image = segmented_image.reshape(image.shape)

# Convert back to BGR for displaying with OpenCV
segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
print(segmented_image_bgr)
cv2.imshow('Segmented Image', segmented_image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()