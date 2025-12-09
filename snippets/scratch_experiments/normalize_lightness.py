import cv2
import numpy as np


def compute_average_lightness(image):
    # Exclude white and black pixels from calculations
    mask = np.all(image != [255, 255, 255], axis=-1) & np.all(image != [0, 0, 0], axis=-1)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0][mask]
    return np.mean(l_channel) if l_channel.size != 0 else 0


def normalize_lightness(image, target_lightness):
    mask = np.all(image != [255, 255, 255], axis=-1) & np.all(image != [0, 0, 0], axis=-1)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l_channel = lab[:, :, 0][mask]
    current_lightness = np.mean(l_channel) if l_channel.size != 0 else 0

    if current_lightness == 0:  # Avoid division by zero
        return image

    # Calculate the adjustment factor
    adjustment_factor = target_lightness / current_lightness

    # Adjust only the non-black and non-white pixels
    lab[mask, 0] = np.clip(l_channel * adjustment_factor, 0, 255)

    # Convert LAB back to BGR
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)

import os

image_folder = "D:/bcc/msr_imgs"
output_folder = "D:/bcc/msr_imgs2"
filenames = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Calculate average lightness over all images
all_lightness = []
for filename in filenames:
    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)
    all_lightness.append(compute_average_lightness(img))

target_lightness = np.mean(all_lightness)

# Normalize and save each image
for filename in filenames:
    img_path = os.path.join(image_folder, filename)
    img = cv2.imread(img_path)
    normalized_img = normalize_lightness(img, target_lightness)
    cv2.imwrite(os.path.join(output_folder, filename), normalized_img)