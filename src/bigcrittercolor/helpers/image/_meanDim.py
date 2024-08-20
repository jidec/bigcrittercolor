import cv2

def _meanDim(images):
    total_width = 0
    total_height = 0
    num_images = len(images)

    for image in images:
        height, width, _ = image.shape
        total_width += width
        total_height += height

    mean_width = int(total_width / num_images)
    mean_height = int(total_height / num_images)

    return mean_width, mean_height