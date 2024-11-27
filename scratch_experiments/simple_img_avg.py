import cv2
import numpy as np
import os


def get_image_paths(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
            os.path.splitext(f)[1].lower() in image_extensions]


def read_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    return images


def compute_average_dimensions(images):
    total_height = 0
    total_width = 0
    num_images = len(images)
    for img in images:
        h, w = img.shape[:2]
        total_height += h
        total_width += w
    avg_height = total_height // num_images
    avg_width = total_width // num_images
    return avg_height, avg_width


def resize_images(images, target_height, target_width):
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, (target_width, target_height))
        resized_images.append(resized_img)
    return resized_images


def compute_average_image(images):
    avg_image = np.mean(images, axis=0).astype(np.uint8)
    return avg_image


def main(folder_path):
    image_paths = get_image_paths(folder_path)
    images = read_images(image_paths)

    if not images:
        print("No images found in the folder.")
        return

    avg_height, avg_width = compute_average_dimensions(images)
    resized_images = resize_images(images, avg_height, avg_width)
    avg_image = compute_average_image(resized_images)

    cv2.imshow("Averaged Image", avg_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Provide the path to your image folder
folder_path = 'D:/bcc/ringtails/segments'
main(folder_path)