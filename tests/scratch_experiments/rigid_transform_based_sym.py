import cv2
import numpy as np

# Load image in grayscale
image = cv2.imread("E:/aeshna_data_appr1/masks/INAT-33906099-1_mask.png", cv2.IMREAD_GRAYSCALE)

# Find keypoints and descriptors with ORB
orb = cv2.ORB_create()
keypoints1, descriptors1 = orb.detectAndCompute(image, None)

# Flip the image and compute keypoints and descriptors for the flipped image
flipped_image = cv2.flip(image, 1)
keypoints2, descriptors2 = orb.detectAndCompute(flipped_image, None)

# Use BFMatcher to find the best matches between the descriptors
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

# Use only the good matches to estimate the rigid transformation
if len(matches) > 4:
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate rigid transformation
    M = cv2.estimateRigidTransform(src_pts, dst_pts, False)

    # If M is None, no transformation is found. Otherwise, continue
    if M is not None:
        # Line of symmetry: y-axis transformed by half this transformation
        center_x = image.shape[1] / 2
        transformed_center = (M @ np.array([center_x, 0, 1]).reshape(-1, 1)).flatten()

        # Draw the line of symmetry on the original image
        cv2.line(image, (int(center_x), 0), (int(transformed_center[0]), image.shape[0]), (0, 0, 255), 2)
        cv2.imshow('Image with Line of Symmetry', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()