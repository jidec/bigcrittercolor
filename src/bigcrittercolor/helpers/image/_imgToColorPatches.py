import numpy as np
import cv2
from bigcrittercolor.helpers import _showImages
from bigcrittercolor.helpers.clustering import _cluster
from bigcrittercolor.helpers.image import _format
from collections import Counter

# from an image, create a new smoothed image that is a series of color patches
# note that blur (typically bilateral) should be done before this function is applied
# this is used in clusterColorsFromPatterns
def _imgToColorPatches(img, bg_mask=None, cluster_args={'n':10, 'algo':'kmeans'}, input_colorspace="rgb", use_median=False, min_patch_pixel_area = 5, return_patch_masks_colors_imgshapes=False, show=False):

    #if input_colorspace is not "rgb":
    #    img = _format._format(img, in_format=input_colorspace,out_format="rgb",alpha=False)
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
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4,
                                                                             ltype=cv2.CV_32S)
        if show:
            img_rgb = _format._format(img, in_format=input_colorspace, out_format="rgb")
            # Map component labels to hue val
            label_hue = np.uint8(179 * labels_im / np.max(labels_im))
            blank_ch = 255 * np.ones_like(label_hue)

            # Create a colored image to visualize the components
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

            # Convert from HSV to BGR for display
            labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

            # Set background label to black
            labeled_img[label_hue == 0] = 0

            # Overlay component numbers
            for i in range(1, num_labels):  # Start from 1 to skip the background
                cv2.putText(labeled_img, str(i), (int(centroids[i][0]), int(centroids[i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)

            #_showImages(show, [mask,labeled_img], titles=["Patch Cluster Mask","Connected Components"])

        for j in range(1, num_labels):  # Start from 1 to skip the background
            if stats[j, cv2.CC_STAT_AREA] >= min_patch_pixel_area:
                # Create a mask for the current component
                component_mask = (labels_im == j).astype(np.uint8) * 255

                if bg_mask is not None:
                    component_mask = component_mask & ~bg_mask

                # Compute the mean color of the pixels in the original image where the current component is located
                mean_color = cv2.mean(img, mask=component_mask)[:3]

                if use_median:
                    masked_pixels = img[component_mask != 0]
                    mean_color = np.median(masked_pixels, axis=0)

                # Apply the mean color to the component in the overlay image
                overlay_image[component_mask == 255] = mean_color

                # _showImages(show, [component_mask,overlay_image], titles=["New Component Mask","Updated Overlay Image"])

                patch_bool_masks.append(component_mask == 255)
                patch_colors.append(mean_color)

    img_rgb = _format._format(img,in_format=input_colorspace,out_format="rgb")
    overlay_img_rgb = _format._format(overlay_image, in_format=input_colorspace, out_format="rgb")
    _showImages(show, [img_rgb, overlay_img_rgb], ['Image', 'Patch Image'])
    #_showImages(show, [img, overlay_image], ['Image', 'Patch Image'])

    if return_patch_masks_colors_imgshapes:
        return((patch_bool_masks,patch_colors,shape))

    return overlay_image

#img = cv2.imread("D:/bcc/beetles/segments/INATRANDOM-16257682_segment.png")
#img = _imgToColorPatches2(img,show=True)