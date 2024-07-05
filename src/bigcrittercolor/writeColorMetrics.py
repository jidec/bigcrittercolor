import os
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from skimage.color import rgb2hsv, rgb2lab

from bigcrittercolor.helpers import _readBCCImgs, _getBCCIDs
from bigcrittercolor.helpers.ids import _imgNameToID, _imgPathToName

def writeColorMetrics(img_ids=None, from_stage="pattern", batch_size=None, pattern_subfolder=None, data_folder=''):

    # get all segment or pattern ids if None
    if img_ids is None:
        img_ids = _getBCCIDs(type=from_stage, data_folder=data_folder)

    all_metrics = []
    if batch_size is None:
        batch_size = len(img_ids)
    def process_batch(batch_img_ids):
        # todo - make sure read works for patterns
        imgs = _readBCCImgs(img_ids=batch_img_ids, data_folder=data_folder)

        if from_stage == "segment":
            simple_metrics = _getSimpleColorMetrics(imgs, batch_img_ids)
            #thresh_metrics = _getThresholdMetrics(imgs, batch_img_ids)
            #metrics = pd.merge(simple_metrics, thresh_metrics, on='img_id')
            metrics = simple_metrics
        if from_stage == "pattern":
            metrics = _getColorClusterMetrics(imgs, batch_img_ids)

        return metrics

    # Iterate over the image IDs in batches
    for i in range(0, len(img_ids), batch_size):
        batch_img_ids = img_ids[i:i + batch_size]
        batch_metrics = process_batch(batch_img_ids)
        all_metrics.append(batch_metrics)
        print(str(i) + "/" + str(len(img_ids)))

    # Concatenate all the batch metrics into a single DataFrame
    metrics = pd.concat(all_metrics, ignore_index=True)

    # TEMP while only doing 1 img per obs
    metrics['obs_id'] = metrics['img_id'].str.replace('-1$', '', regex=True)

    metrics.to_csv(data_folder + "/metrics.csv",index=False)
    records = pd.read_csv(data_folder + "/records.csv")

    # TEMP while only doing 1 img per obs
    records_with_metrics = pd.merge(metrics,records,on='obs_id')

    records_with_metrics.to_csv(data_folder + "/records_with_metrics.csv",index=False)

# means, thresholds
def _getSimpleColorMetrics(imgs, img_ids):
    data = []

    for img, img_id in zip(imgs, img_ids):

        image = img
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mask to exclude black pixels
        mask = (image_rgb[:, :, 0] != 0) | (image_rgb[:, :, 1] != 0) | (image_rgb[:, :, 2] != 0)

        if not np.any(mask):
            # Skip if no non-black pixels
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

def _getThresholdMetrics(segs, img_ids, thresh_values=[0.1]):
    data = []

    for img, img_id in zip(segs, img_ids):

        image = img
        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Mask to exclude black pixels
        mask = (image_rgb[:, :, 0] != 0) | (image_rgb[:, :, 1] != 0) | (image_rgb[:, :, 2] != 0)

        if not np.any(mask):
            # Skip if no non-black pixels
            continue

        # Apply mask
        image_rgb = image_rgb[mask]

        # Convert to HSV
        hsv_image = rgb2hsv(image_rgb.reshape(-1, 1, 3) / 255.0)

        # Extract the value channel
        value_channel = hsv_image[:, :, 2]

        # Initialize the data dictionary for this image
        image_data = {'img_id': img_id}

        # Calculate the percentage for each threshold value
        for thresh in thresh_values:
            dark_pixels_percent = np.mean(value_channel <= thresh)
            image_data[f'percent_dark_or_darker_than_{thresh}'] = dark_pixels_percent * 100

        # Append the metrics to the data list
        data.append(image_data)

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)
    return df


def _getColorClusterMetrics(images, img_ids):
    # Loop through first to find all unique colors
    all_colors = set()
    for i, image in enumerate(images):
        if i % 100 == 0:
            print(i)
        arr = np.array(image)
        # Remove black background pixels and flatten to list of colors
        arr = arr[(arr[:, :, 0] != 0) | (arr[:, :, 1] != 0) | (arr[:, :, 2] != 0)]
        for color in np.unique(arr, axis=0):
            all_colors.add(tuple(color))

    uniq_colors = np.array(list(all_colors))

    # Loop through again to get proportions and build dataframe
    data = []
    for i, (image, img_id) in enumerate(zip(images, img_ids)):
        if i % 100 == 0:
            print(i)
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

#writeColorMetrics(from_stage="segment",data_folder="D:/bcc/ringtails")