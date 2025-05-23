import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2hsv, rgb2lab
import matplotlib.pyplot as plt

from bigcrittercolor.helpers import _readBCCImgs, _getBCCIDs, _showImages, makeCollage, _bprint
from bigcrittercolor.helpers.ids import _imgNameToID, _imgPathToName
from bigcrittercolor.helpers.image import _format

def writeBasicColorMetrics(img_ids=None, from_stage="segment", batch_size=1000,
                      get_color_metrics=True,
                      get_shape_texture_metrics=True,
                      threshold_metrics=[("hls",2,0.3,"below"),("rgb",0,0.6,"above")],
                      pattern_subfolder=None, show=False, print_steps=True, data_folder=''):

    """ Write color metrics like mean colors and thresholds directly obtained from filtered segments or color-clustered "patterns".

        Metrics appended to records are written to *data_folder*/records_with_metrics.csv.

        Args:
            from_stage (str): either "segment" or "pattern" (pattern referring to color clustered pattern)
            batch_size (int): get metrics in batches of this size - use to conserve memory when running tens or hundreds of thousands of images
            simple_metrics (bool): whether to get a suite of simple metrics like color channel means
            thresh_metrics (list):  a list of tuples defining the threshold metrics. The first element of each tuple is the colorspace (either "hls" or "rgb), the second element is the channel (0,1, or 2), the third element is the threshold value, and the fourth element is "above" or "below" to measure the percent area either above or below the threshold
    """

    _bprint(print_steps, "Starting writeSimpleColorMetrics...")
    # get all segment or pattern ids if None
    if img_ids is None:
        _bprint(print_steps, "No image ids specified, getting ids from all segments...")
        img_ids = _getBCCIDs(type=from_stage, data_folder=data_folder)

    # list to hold metrics
    all_metrics = []

    # do this in batches to avoid loading all images at once
    if batch_size is None:
        batch_size = len(img_ids)
    def process_batch(batch_img_ids, from_stage, show):
        # todo - make sure read works for patterns
        imgs = _readBCCImgs(type=from_stage,img_ids=batch_img_ids, data_folder=data_folder)

        if from_stage == "segment":
            simple_metrics = _getSimpleColorMetrics(imgs, batch_img_ids)
            thresh_metrics = _getThresholdMetrics(imgs, batch_img_ids, thresholds=threshold_metrics, show=show)
            shape_tex_metrics = _getShapeTextureMetrics(imgs,batch_img_ids,show=show)
            metrics = pd.merge(simple_metrics, thresh_metrics, on='img_id')
            metrics = pd.merge(metrics, shape_tex_metrics, on='img_id')
            #metrics = simple_metrics
        if from_stage == "pattern":
            metrics = _getColorClusterMetrics(imgs, batch_img_ids)

        return metrics

    _bprint(print_steps, "Obtaining metrics in batches of size " + str(batch_size) + "...")
    # iterate over the image IDs in batches
    for i in range(0, len(img_ids), batch_size):
        batch_img_ids = img_ids[i:i + batch_size]
        batch_metrics = process_batch(batch_img_ids,from_stage,show)
        all_metrics.append(batch_metrics)
        print(str(i) + "/" + str(len(img_ids)))

    _bprint(print_steps, "Cleaning metric dataframe...")
    # concatenate all the batch metrics into a single DataFrame
    metrics = pd.concat(all_metrics, ignore_index=True)

    # while only doing 1 img per obs (temporary)
    metrics['obs_id'] = metrics['img_id'].str.replace('-1$', '', regex=True)

    metrics.to_csv(data_folder + "/metrics.csv",index=False)
    records = pd.read_csv(data_folder + "/records.csv")

    # while only doing 1 img per obs (temporary)
    records_with_metrics = pd.merge(metrics,records,on='obs_id')

    # Rename the column 'img_id_x' to 'img_id'
    records_with_metrics.rename(columns={'img_id_x': 'img_id'}, inplace=True)
    # Remove the column 'img_id_y'
    records_with_metrics.drop(columns=['img_id_y'], inplace=True)

    _bprint(print_steps, "Writing metric dataframe to " + data_folder + "/records_with_metrics.csv" + "...")
    records_with_metrics.to_csv(data_folder + "/records_with_metrics.csv",index=False)
    _bprint(print_steps, "Finished writeSimpleColorMetrics.")

# means, thresholds
def _getSimpleColorMetrics(imgs, img_ids):
    data = []

    for img, img_id in zip(imgs, img_ids):

        image = img
        # convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask to exclude black pixels
        mask = (image_rgb[:, :, 0] != 0) | (image_rgb[:, :, 1] != 0) | (image_rgb[:, :, 2] != 0)

        if not np.any(mask):
            # skip if no non-black pixels
            continue

        # Apply mask
        image_rgb = image_rgb[mask]

        # Compute mean RGB
        mean_red = np.mean(image_rgb[:, 0])
        mean_green = np.mean(image_rgb[:, 1])
        mean_blue = np.mean(image_rgb[:, 2])

        # Convert to HSV and compute mean hue and saturation
        hsv_image = rgb2hsv(image_rgb.reshape(-1, 1, 3) / 255.0)
        mean_hue = np.mean(hsv_image[:, :, 0])
        mean_saturation = np.mean(hsv_image[:, :, 1])
        mean_value = np.mean(hsv_image[:, :, 2])

        # Convert to CIELAB and compute mean lightness
        lab_image = rgb2lab(image_rgb.reshape(-1, 1, 3) / 255.0)
        mean_cielab_lightness = np.mean(lab_image[:, :, 0])

        # Append the metrics to the data list
        data.append({
            'img_id': img_id,
            'mean_red': mean_red,
            'mean_green': mean_green,
            'mean_blue': mean_blue,
            'mean_hue': mean_hue,
            'mean_saturation': mean_saturation,
            'mean_value': mean_value,
            'mean_cielab_lightness': mean_cielab_lightness
        })

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    return df

def _getThresholdMetrics(segs, img_ids, thresholds=[("hls", 2, 0.3)], show=False):
    data = []
    imgs_to_show = []

    for img, img_id in zip(segs, img_ids):
        image = img
        # convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # mask to exclude black pixels
        mask = (image_rgb[:, :, 0] != 0) | (image_rgb[:, :, 1] != 0) | (image_rgb[:, :, 2] != 0)

        # initialize the data dictionary for this image
        image_data = {'img_id': img_id}

        for colorspace, channel, thresh, below_or_above in thresholds:
            formatted_img = _format(image_rgb, in_format="rgb", out_format=colorspace)
            channel_img = formatted_img[:, :, int(channel)]

            # if below, count pixels below the threshold
            if below_or_above == "below":
                thresh_pixels_percent = np.mean(channel_img[mask] <= float(thresh))
            # otherwise count above the threshold
            elif below_or_above == "above":
                thresh_pixels_percent = np.mean(channel_img[mask] >= float(thresh))
            image_data[f'{colorspace}_channel{channel}_thresh{thresh}_{below_or_above}'] = thresh_pixels_percent * 100

            if show:
                below_thresh_mask = channel_img <= thresh
                marked_image_bgr = image_rgb.copy()
                marked_image_bgr = cv2.cvtColor(marked_image_bgr, cv2.COLOR_RGB2BGR)
                marked_image_bgr[below_thresh_mask & mask] = [0, 0, 255]  # Mark in red
                thresh_collage = makeCollage([image, marked_image_bgr], n_per_row=2)
                imgs_to_show.append(thresh_collage)

        # Append the metrics to the data list
        data.append(image_data)

    _showImages(show, imgs_to_show, sample_n=18)

    df = pd.DataFrame(data)
    return df


from skimage.measure import regionprops_table, label
from skimage.filters import gabor
from scipy.stats import kurtosis, entropy

def _getShapeTextureMetrics(imgs, img_ids, show=False):
    data = []

    for img, img_id in zip(imgs, img_ids):
        # Ensure image is grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Binarize the image (assuming foreground is non-black)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        _showImages(show,images=[binary],maintitle="Binary")

        # Label the binary regions to extract connected components
        labeled_img = label(binary)
        props = regionprops_table(labeled_img, properties=[
            'major_axis_length', 'minor_axis_length', 'area', 'solidity', 'eccentricity'
        ])

        if len(props['area']) == 0:  # Skip if no objects detected
            continue

        # Extract shape metrics
        length = props['major_axis_length'][0]
        width = props['minor_axis_length'][0]
        area = props['area'][0]
        solidity = props['solidity'][0]
        eccentricity = props['eccentricity'][0]
        length_width_ratio = length / width if width > 0 else None

        # Circularity: 4π * Area / Perimeter²
        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Check if any contours exist
        if len(contours) > 0:
            # Take the largest contour (or the first contour as default)
            largest_contour = max(contours, key=cv2.contourArea)

            # Compute the perimeter of the largest contour
            perimeter = cv2.arcLength(largest_contour, True)

            # Visualize the contour
            # Create a copy of the original binary image to draw on
            contour_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  # Convert to color for visualization
            cv2.drawContours(contour_image, [largest_contour], -1, (0, 255, 0), 2)  # Green contour, thickness 2
            _showImages(show,images=[contour_image],maintitle="Contour")
        else:
            # No contours found; set perimeter to None or 0
            perimeter = 0
        #perimeter = cv2.arcLength(cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else None

        # Compute Hu Moments
        moments = cv2.moments(binary)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu1, hu2, hu3, hu4 = hu_moments[:4]

        # Texture metrics
        non_zero_pixels = gray[gray > 0]
        entropy_value = entropy(np.histogram(non_zero_pixels, bins=256, density=True)[0])
        kurtosis_value = kurtosis(non_zero_pixels, axis=None)

        # Gabor filter responses
        gabor_horizontal = gabor(gray, frequency=0.2, theta=0)  # Horizontal: theta=0
        gabor_vertical = gabor(gray, frequency=0.2, theta=np.pi / 2)  # Vertical: theta=90°

        # Mean Gabor responses
        gabor_horizontal_mean = np.mean(np.abs(gabor_horizontal[0]))
        gabor_vertical_mean = np.mean(np.abs(gabor_vertical[0]))

        # Append to data
        data.append({
            'img_id': img_id,
            'length_pixels': length,
            'width_pixels': width,
            'area_pixels': area,
            'length_width_ratio': length_width_ratio,
            'circularity': circularity,
            'eccentricity_elongation': eccentricity,
            'solidity_nonconcavity': solidity,
            'hu1_xy_spread_size': hu1,
            'hu2_xy_asymmetry': hu2,
            'hu3_diagonal_asymmetry': hu3,
            'hu4_elongation': hu4,
            'entropy_irregularity': entropy_value,
            'kurtosis_dark_bright_tails': kurtosis_value,
            'gabor_horizontal': gabor_horizontal_mean,
            'gabor_vertical': gabor_vertical_mean
        })

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

def _getColorClusterMetrics(images, img_ids):
    # Loop through first to find all unique colors
    all_colors = set()
    for i, image in enumerate(images):
        #if i % 100 == 0:
        #    print(i)
        arr = np.array(image)
        # Remove black background pixels and flatten to list of colors
        arr = arr[(arr[:, :, 0] != 0) | (arr[:, :, 1] != 0) | (arr[:, :, 2] != 0)]
        for color in np.unique(arr, axis=0):
            all_colors.add(tuple(color))

    uniq_colors = np.array(list(all_colors))

    # Loop through again to get proportions and build dataframe
    data = []
    for i, (image, img_id) in enumerate(zip(images, img_ids)):
        #if i % 100 == 0:
        #    print(i)
        arr = np.array(image)
        arr = arr[(arr[:, :, 0] != 0) | (arr[:, :, 1] != 0) | (arr[:, :, 2] != 0)]

        # Calculate proportion of pixels for each unique color
        row = [img_id]
        total_pix = arr.shape[0]
        for color in uniq_colors:
            col_pix = np.sum((arr[:, :3] == color[:3]).all(axis=1))
            prop = col_pix / total_pix if total_pix else 0
            row.extend(list(color[:3]) + [prop])

        data.append(row)

    # Creating DataFrame
    columns = ['img_id']
    for i, color in enumerate(uniq_colors, start=1):
        columns.extend([f'col_{i}_r', f'col_{i}_g', f'col_{i}_b', f'col_{i}_prop'])

    df = pd.DataFrame(data, columns=columns)
    return df

#writeBasicColorMetrics(from_stage="segment",data_folder="D:/bcc/ringtails",
#                  threshold_metrics=[("hls",1,83,"below")],show=True)
#("rgb",1,125,"above")