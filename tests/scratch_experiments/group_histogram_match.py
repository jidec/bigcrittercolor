import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def calculate_master_histogram(images):
    master_histogram = np.zeros(256)
    for img in images:
        for channel in cv2.split(img):
            hist, _ = np.histogram(channel, bins=256, range=(0, 256))
            master_histogram += hist
    master_histogram /= master_histogram.sum()  # Ensure normalization
    return master_histogram

def match_histograms(images, master_histogram):
    matched_images = []
    master_cdf = np.cumsum(master_histogram)
    master_cdf = (255 * (master_cdf / master_cdf[-1])).astype(np.uint8)  # Normalize and convert to proper scale

    for img in images:
        matched_img = np.zeros_like(img)
        for i, channel in enumerate(cv2.split(img)):
            hist, bins = np.histogram(channel.flatten(), bins=256, range=[0,256])
            cdf = np.cumsum(hist)
            cdf = (255 * (cdf / cdf[-1])).astype(np.uint8)  # Normalize and convert to proper scale

            # Use np.interp for interpolation of pixel values
            im2 = np.interp(channel.flatten(), bins[:-1], cdf)
            matched_channel = np.interp(im2, master_cdf, np.arange(256))
            matched_img[:,:,i] = matched_channel.reshape(channel.shape)
        matched_images.append(matched_img)
    return matched_images

def visualize_images(original_images, matched_images):
    for original, matched in zip(original_images, matched_images):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(matched, cv2.COLOR_BGR2RGB))
        plt.title('Matched Image')
        plt.axis('off')

        plt.show()

# Load images
folder_path = 'D:/bcc/msr_imgs'
images = load_images_from_folder(folder_path)

# Calculate the master histogram
master_histogram = calculate_master_histogram(images)

# Match histograms of all images to the master histogram
matched_images = match_histograms(images, master_histogram)

# Visualize the original and matched images side by side
visualize_images(images, matched_images)