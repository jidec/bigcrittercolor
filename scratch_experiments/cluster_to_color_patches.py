import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the image
#image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4988619_segment.png')
image1 = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-4263361_segment.png')
#image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-5618336_segment.png')
#image = cv2.imread('D:/bcc/dfly_appr_expr/appr3/segments/INATRANDOM-38377437_segment.png')
image2 = cv2.imread('E:/aeshna_data/segments/INAT-130944-1_segment.png')
image3 = cv2.imread('E:/aeshna_data/segments/INAT-7444110-10_segment.png')
#image = cv2.imread('E:/aeshna_data/segments/INAT-7088777-1_segment.png')
image4 = cv2.imread('E:/aeshna_data/segments/INAT-7656493-4_segment.png')

imgs = [image1,image2,image3,image4]

patch_imgs = []
for image in imgs:
    original_shape = image.shape[:2]  # height, width

    # Apply bilateral filter to smooth out the image
    filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    cv2.imshow("0",filtered_image)
    cv2.waitKey(0)

    # Reshape the image to a 2D array of pixels
    pixel_values = filtered_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5  # Number of clusters
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Map the centers to the labels to get the segmented image
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image shape
    segmented_image = segmented_image.reshape(image.shape)  # Make sure this matches the original image's shape

    # Define a minimum area threshold
    min_area_threshold = 50  # Adjust this value as needed

    # Copy the original image for overlaying patches
    overlay_image = image.copy()

    # Process each cluster and overlay patches
    for i in range(k):
        # Create a mask for the current cluster
        mask = (labels.reshape(original_shape) == i).astype(np.uint8) * 255

        # Find connected components in the mask
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)

        for j in range(1, num_labels):  # Start from 1 to skip the background
            if stats[j, cv2.CC_STAT_AREA] >= min_area_threshold:
                # Create a mask for the current component
                component_mask = (labels_im == j).astype(np.uint8) * 255
                # Compute the mean color of the pixels in the original image where the current component is located
                mean_color = cv2.mean(image, mask=component_mask)[:3]

                masked_pixels = image[component_mask != 0]

                # Compute the median color
                # Since masked_pixels will have a shape of (N, 3) for a color image,
                # where N is the number of non-zero pixels in the mask, we compute the median along the 0th axis.
                #mean_color = np.median(masked_pixels, axis=0)

                # Apply the mean color to the component in the overlay image
                overlay_image[component_mask == 255] = mean_color

    # Display the original image with the color patches overlayed
    plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image with Mean Color Patches')
    plt.axis('off')
    plt.show()

    patch_imgs.append(overlay_image)

from bigcrittercolor.helpers import _showImages

_showImages(True,imgs + patch_imgs,num_cols=4)