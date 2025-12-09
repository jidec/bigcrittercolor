import cv2
import numpy as np

# Read the images
image1 = cv2.imread('D:/bcc/ringtails/segments/INAT-396575-1_segment.png')
image2 = cv2.imread('D:/bcc/ringtails/segments/INAT-688803-1_segment.png')

cv2.imshow("0",image2)
cv2.waitKey(0)
# Detect ORB features and compute descriptors
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Check if descriptors are valid
if descriptors1 is None or descriptors2 is None:
    raise ValueError("One of the descriptor sets is empty. Please check the input images.")

# Match features using Brute-Force matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Ensure the descriptors are of the same type
if descriptors1.dtype == descriptors2.dtype and descriptors1.shape[1] == descriptors2.shape[1]:
    matches = bf.match(descriptors1, descriptors2)
else:
    raise ValueError("Descriptor type or size mismatch between the two images.")

# Sort matches by distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Extract the matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Estimate affine transformation matrix
matrix, mask = cv2.estimateAffine2D(src_pts, dst_pts)

# Apply the transformation to align the second image to the first
aligned_image2 = cv2.warpAffine(image2, matrix, (image1.shape[1], image1.shape[0]))

# Compute the average image
average_image = cv2.addWeighted(image1, 0.5, aligned_image2, 0.5, 0)

# Save or display the result
cv2.imwrite('average_image.png', average_image)
cv2.imshow('Average Image', average_image)
cv2.waitKey(0)
cv2.destroyAllWindows()