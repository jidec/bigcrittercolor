import numpy as np
import cv2
from bigcrittercolor.helpers import _showImages
from bigcrittercolor.helpers.clustering import _cluster
from collections import Counter

# from an image, create a new smoothed image that is a series of color patches
# note that blur (typically bilateral) should be done before this function is applied
# this is used in clusterColorsFromPatterns
def _imgToColorPatches(img, cluster_args={'n':4, 'algo':'kmeans'}, input_colorspace="cielab", use_median=False, min_patch_pixels = 5, return_patch_masks_colors_imgshapes=False, show=False):

    shape = img.shape
    original_shape = img.shape[:2]  # height, width

    # Reshape the image to a 2D array of pixels
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Call the _cluster function instead of cv2.kmeans
    labels = _cluster(pixel_values, **cluster_args)

    # Convert labels to a 1D array if necessary
    labels = labels.flatten() if labels.ndim > 1 else labels

    # Copy the original image for overlaying patches
    overlay_image = img.copy()
    patch_bool_masks = []
    patch_colors = []

    # Process each cluster and overlay patches
    for label in np.unique(labels):
        # Create a mask for the current cluster
        mask = (labels.reshape(original_shape) == label).astype(np.uint8) * 255

        # Find connected components in the mask
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8,
                                                                                   ltype=cv2.CV_32S)

        for j in range(1, num_labels):  # Start from 1 to skip the background
            if stats[j, cv2.CC_STAT_AREA] >= min_patch_pixels:
                # Create a mask for the current component
                component_mask = (labels_im == j).astype(np.uint8) * 255

                # The below removes pure black background pixels from masks
                # Apply the mask to the original image
                masked_img = cv2.bitwise_and(img, img, mask=component_mask)
                # Check for black pixels
                target_black_value = [0,0,0]
                if input_colorspace == "cielab":
                    target_black_value = [10,128,128]
                black_pixels = np.all(masked_img <= target_black_value, axis=-1)
                component_mask[black_pixels] = 0  # Remove black pixel locations from the mask

                # Compute the mean color of the pixels in the original image where the current component is located
                mean_color = cv2.mean(img, mask=component_mask)[:3]

                if use_median:
                    masked_pixels = img[component_mask != 0]
                    mean_color = np.median(masked_pixels, axis=0)
                # Compute the median color
                # Since masked_pixels will have a shape of (N, 3) for a color image,
                # where N is the number of non-zero pixels in the mask, we compute the median along the 0th axis.

                # Apply the mean color to the component in the overlay image
                overlay_image[component_mask == 255] = mean_color

                if mean_color[0] > 10:
                    patch_bool_masks.append(component_mask == 255)
                    patch_colors.append(mean_color)

    _showImages(show, [img, overlay_image], ['Image', 'Patch Image'])

    if return_patch_masks_colors_imgshapes:
        return((patch_bool_masks,patch_colors,shape))

    return overlay_image

#img = cv2.imread("D:/bcc/beetles/segments/INATRANDOM-16257682_segment.png")
#img = _imgToColorPatches2(img,show=True)