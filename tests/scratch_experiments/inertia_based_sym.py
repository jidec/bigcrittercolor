import cv2
import numpy as np

# Load image in grayscale
image = cv2.imread("E:/aeshna_data_appr1/masks/INAT-33906099-1_mask.png", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

#image = cv2.resize()
# Binarize the image: consider pixels with intensity > 128 as white
_, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

# Find the coordinates of the white pixels
y, x = np.where(binary_image == 255)

# Compute the centroid of the white pixels
centroid = (x.mean(), y.mean())

# Translate (center) the coordinates
x_centered = x - centroid[0]
y_centered = y - centroid[1]

# Compute the components of the inertia tensor
Ixx = np.sum(y_centered**2)
Iyy = np.sum(x_centered**2)
Ixy = -np.sum(x_centered*y_centered)

inertia_tensor = np.array([
    [Ixx, Ixy],
    [Ixy, Iyy]
])

# Compute eigenvectors and eigenvalues
eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)

# The rotation angle can be found from the arctangent of the eigenvectors
angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

# Convert the angle from radians to degrees
angle_deg = np.degrees(angle)

print(f"Rotation angle (in degrees): {angle_deg}")

# If you want to visualize the principal axes on the original image:
height, width = image.shape
center = (int(centroid[0]), int(centroid[1]))
cv2.line(image, center, (int(center[0] + width * eigenvectors[0, 0]), int(center[1] + height * eigenvectors[1, 0])), (0, 0, 255), 2)
cv2.line(image, center, (int(center[0] + width * eigenvectors[0, 1]), int(center[1] + height * eigenvectors[1, 1])), (0, 0, 255), 4)
cv2.imshow('Image with Principal Axes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()