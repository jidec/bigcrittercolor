import cv2
import numpy as np
import os
import glob

def single_scale_retinex(image, sigma):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    retinex = cv2.log(image + 1.0) - cv2.log(blurred + 1.0)
    return retinex

def multi_scale_retinex(image, sigmas):
    retinex = np.zeros_like(image)
    for sigma in sigmas:
        retinex += single_scale_retinex(image, sigma)
    retinex /= len(sigmas)
    return retinex


def color_balance(image, percent=1):
    out_channels = []
    for channel in cv2.split(image):
        # Calculate the histogram
        hist = cv2.calcHist([channel], [0], None, [256], (0, 256))
        cumhist = np.cumsum(hist)

        # Calculate the cut-off points
        low_cut, high_cut = np.percentile(channel, (percent, 100 - percent))

        # Find the minimum and maximum pixel intensities that are not outliers
        v_min = np.searchsorted(cumhist, low_cut)
        v_max = np.searchsorted(cumhist, high_cut)

        # Apply contrast stretching to the channel
        channel = np.clip((channel - v_min) * 255 / (v_max - v_min), 0, 255).astype('uint8')

        out_channels.append(channel)

    return cv2.merge(out_channels)

def estimate_illumination(image, sigmas):
    return np.exp(multi_scale_retinex(image, sigmas))

def reconstruct_image(reflectance, illumination):
    return cv2.multiply(reflectance, illumination)

def load_and_resize_images(image_paths, target_size):
    images = []
    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Unable to load image at path {path}")
            continue
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        images.append(image)
    return images

# Define your scales - these are arbitrary and should be tuned for your specific images
sigmas = [15, 80, 250]

# Load all images
image_folder = 'D:/bcc/msr_imgs'
image_paths = glob.glob(os.path.join(image_folder, '*.png'))

# Determine target size from the first image (or set manually)
first_image = cv2.imread(image_paths[0])
if first_image is None:
    raise ValueError("Error loading the first image.")
target_size = (first_image.shape[1], first_image.shape[0])

images = load_and_resize_images(image_paths, target_size)

# Convert to float and scale to [0, 1]
images = [np.float32(img) / 255 for img in images]

# Estimate the illumination for each image
illuminations = [estimate_illumination(img, sigmas) for img in images]

# Calculate the average illumination
average_illumination = np.mean(np.array(illuminations), axis=0)

# Normalize the illumination of each image and reconstruct
# Normalize the illumination of each image and reconstruct
normalized_images = []
for img, ill in zip(images, illuminations):
    normalized_ill = ill / average_illumination
    reflectance = img / ill
    normalized_img = reconstruct_image(reflectance, normalized_ill)

    # Ensure pixel values are in the range [0, 255]
    normalized_img = np.clip(normalized_img * 255, 0, 255).astype('uint8')

    # Apply color balance
    #normalized_img = color_balance(normalized_img)
    normalized_images.append(normalized_img)

# Save or display the results
for i, img in enumerate(normalized_images):
    cv2.imwrite(os.path.join(image_folder, f'normalized_{i}.jpg'), img)
    cv2.imshow(f'Normalized Image {i}', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()