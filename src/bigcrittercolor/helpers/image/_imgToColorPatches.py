import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, deltaE_cie76
from bigcrittercolor.helpers.image import _blur, _format
from bigcrittercolor.helpers.image import _resizeImgToTotalDim, _reconstructImgFromPPD
from bigcrittercolor.helpers import _showImages

def _imgToColorPatches(img, id, bg_mask=None, input_colorspace="rgb", min_patch_pixel_area=5,
                       use_patch_positions=False,
                       n_seeds=40, region_threshold=20, show=False):
    shape = img.shape
    original_shape = img.shape[:2]  # height, width

    # Initialize variables
    patch_bool_masks = []
    patch_colors = []
    patch_xys = []

    rgb_img = _format._format(img,in_format=input_colorspace,out_format="rgb")
    regions = _getRegionGrowRegions(rgb_img, n_seeds=n_seeds,initial_threshold=region_threshold)

    for region in regions:
        if len(region) >= min_patch_pixel_area:
            component_mask = np.zeros(original_shape, dtype=np.uint8)
            for (x, y) in region:
                component_mask[y, x] = 255

            if bg_mask is not None:
                component_mask = component_mask & ~bg_mask

            if show:
                cv2.imshow("0",component_mask)
                cv2.waitKey(0)
            mean_color = cv2.mean(img, mask=component_mask)[:3]

            if mean_color[0] + mean_color[1] + mean_color[2] < 10:
                continue

            if use_patch_positions:
                coordinates = np.argwhere(component_mask)
                mask_y, mask_x = np.mean(coordinates, axis=0)
                #print((mask_x, mask_y))
                mean_color = mean_color + (mask_x, mask_y)
                patch_xys.append((mask_x, mask_y))
            #if input_colorspace == "cielab":
            #    if np.sum(mean_color) < 10:
            #        continue

            patch_bool_masks.append(component_mask == 255)
            patch_colors.append(mean_color)

    return (patch_bool_masks, patch_colors, shape, id, patch_xys)

# region grow regions takes an RGB image
def _getRegionGrowRegions(image, n_seeds, initial_threshold=10,show=False):
    height, width, channels = image.shape
    processed = np.zeros((height, width), dtype=bool)
    regions = []

    grid_size = max(height, width) // n_seeds
    seeds = [(x, y) for x in range(0, width, grid_size) for y in range(0, height, grid_size)]
    def get_neighbors(x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
        return neighbors

    def color_distance(c1, c2):
        # Use deltaE_cie76 for better color difference measurement
        lab1 = rgb2lab(np.uint8([[c1]]))
        lab2 = rgb2lab(np.uint8([[c2]]))
        return deltaE_cie76(lab1, lab2)[0][0]

    for seed in seeds:
        x, y = seed
        if processed[y, x]:
            continue
        region = []
        stack = [(x, y)]
        seed_color = image[y, x]
        while stack:
            px, py = stack.pop()
            if processed[py, px]:
                continue
            processed[py, px] = True
            region.append((px, py))
            current_threshold = initial_threshold
            for nx, ny in get_neighbors(px, py):
                if not processed[ny, nx]:
                    neighbor_color = image[ny, nx]
                    if not np.array_equal(neighbor_color, [0, 0, 0]):  # Exclude black background
                        if color_distance(neighbor_color, seed_color) < current_threshold:
                            stack.append((nx, ny))
                        else:
                            # If not similar enough, reduce the threshold to capture fine details
                            current_threshold = initial_threshold / 2
                            if color_distance(neighbor_color, seed_color) < current_threshold:
                                stack.append((nx, ny))
        if region:
            regions.append(region)

    if show:
        # Visualize the regions with their mean colors
        output_image = np.zeros_like(image)
        for region in regions:
            # Calculate the mean color of the region
            region_pixels = np.array([image[y, x] for (x, y) in region])
            mean_color = np.mean(region_pixels, axis=0).astype(int)
            for (x, y) in region:
                output_image[y, x] = mean_color
        _showImages(show, [image, output_image], titles=["Original Image", "Region Growing Result"])

    return regions


# # Load the image
# image_path = 'D:/bcc/ringtails/segments/INAT-147315-1_segment.png'
# image_path = 'D:/bcc/ringtails/segments/INAT-150591-1_segment.png'
# image_path = 'D:/bcc/ringtails/segments/INAT-144691-1_segment.png'
# image_path = 'D:/bcc/ringtails/segments/INAT-147316-1_segment.png'
#
# image = cv2.imread(image_path)
# image = _blur._blur(image, type="bilateral")
# #image = _getRegionGrowRegions(image,n_seeds=40,initial_threshold=20,show=True)
#
# ppd = _imgToColorPatches(image,id=1,show=False)
# reconstruct = _reconstructImgFromPPD._reconstructImgFromPPD(ppd)
# _showImages(True,[reconstruct])