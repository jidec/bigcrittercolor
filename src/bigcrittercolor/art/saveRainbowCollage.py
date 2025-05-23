import cv2
import numpy as np

def saveRainbowCollage(images, output_path, collage_width, h_overlap=0, v_overlap=0):
    images = resize_images_to_average(images)
    images = sort_images_by_hue(images)

    img_height, img_width, _ = images[0].shape

    stride_x = int(img_width * (1 - h_overlap))
    stride_y = int(img_height * (1 - v_overlap))

    collage_width_pixels = stride_x * collage_width
    collage_height_pixels = stride_y * ((len(images) + collage_width - 1) // collage_width)

    collage = np.zeros((collage_height_pixels, collage_width_pixels, 3), dtype=np.uint8)

    x_offset = 0
    y_offset = 0

    for idx, img in enumerate(images):
        collage[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img
        x_offset += stride_x

        if (idx + 1) % collage_width == 0:
            x_offset = 0
            y_offset += stride_y

    cv2.imwrite(output_path, collage)

def average_hue(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Mask out black (transparent) areas
    mask = (image[:, :, 0] != 0) | (image[:, :, 1] != 0) | (image[:, :, 2] != 0)
    pixels = image[mask]

    if len(pixels) == 0:  # Handle empty images
        return 0

    # Convert RGB to HSV and calculate average hue
    hsv_pixels = cv2.cvtColor(np.uint8([pixels]), cv2.COLOR_RGB2HSV)[0]
    average_hue = np.mean(hsv_pixels[:, 0])  # Hue channel
    return average_hue


def sort_images_by_hue(images):
    hues = [(image, average_hue(image)) for image in images]
    sorted_images = [img[0] for img in sorted(hues, key=lambda x: x[1])]
    return sorted_images


def resize_images_to_average(images):
    avg_height = int(np.mean([img.shape[0] for img in images]))
    avg_width = int(np.mean([img.shape[1] for img in images]))

    resized_images = [
        cv2.resize(img, (avg_width, avg_height), interpolation=cv2.INTER_AREA) for img in images
    ]
    return resized_images

# Example usage:
#from bigcrittercolor.helpers import _readBCCImgs

#cv2_images_list = _readBCCImgs(type="segment", data_folder="D:/bcc/ringtails")
#saveRainbowCollage(cv2_images_list, "output_collage.jpg", collage_width=100)
