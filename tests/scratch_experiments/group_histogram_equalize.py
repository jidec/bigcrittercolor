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

def equalize_histograms(images):
    sum_histograms = [np.zeros(256) for _ in range(3)]
    for img in images:
        mask = np.all(img > 0, axis=2).astype(np.uint8)
        for i in range(3):
            hist = cv2.calcHist([img], [i], mask, [256], [0,256])
            sum_histograms[i] += hist.flatten()

    equalized_images = []
    for img in images:
        channels = cv2.split(img)
        equalized_channels = []
        for i in range(3):
            sum_histograms[i] /= sum_histograms[i].sum()
            cdf = np.cumsum(sum_histograms[i]) * 255
            cdf_normalized = np.uint8(cdf)
            equalized_channels.append(cv2.LUT(channels[i], cdf_normalized))
        equalized_images.append(cv2.merge(equalized_channels))
    return equalized_images

# Load images
folder_path = 'D:/bcc/msr_imgs'
images = load_images_from_folder(folder_path)

# Equalize images
equalized_images = equalize_histograms(images)

# Display original and equalized images side by side
for original, equalized in zip(images, equalized_images):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
    plt.title('Equalized Image')
    plt.axis('off')

    plt.show()