import numpy as np
import cv2
from PIL import Image, ImageOps
import math
from bigcrittercolor.helpers.image import _resizeImgsToAvgSize

def saveWhorlCollage(cv2_images, output_path, canvas_size=(5000, 5000), center=(2500, 2500), radius_increment=300, scaling_factor=0.1, background_color=(0, 0, 0, 0)):
    """
    Create a whorl collage from a list of OpenCV images with dynamic image distribution per circle.

    Args:
        cv2_images (list): List of OpenCV images (NumPy arrays).
        output_path (str): Path to save the resulting collage.
        canvas_size (tuple): Size of the output canvas (width, height).
        center (tuple): Center of the whorl (x, y).
        radius_increment (int): Increment for radius of each concentric circle.
        scaling_factor (float): Factor to adjust the number of images per circle based on radius.
        background_color (tuple): Background color in RGBA (default: white with transparency).

    Returns:
        None: Saves the resulting collage to the specified output path.
    """

    cv2_images = _resizeImgsToAvgSize(cv2_images)

    # Create a blank canvas with the specified background color
    collage = Image.new("RGBA", canvas_size, background_color)

    num_images = len(cv2_images)
    image_index = 0

    # Loop until all images are placed
    layer = 0
    while image_index < num_images:
        radius = radius_increment * (layer + 1)  # Radius increases for each layer
        # Dynamically calculate the number of images for this layer
        images_per_layer = max(5, int(scaling_factor * radius))  # Ensure a minimum of 5 images per layer
        angle_increment = 360 / images_per_layer  # Divide 360 degrees equally among the images

        for i in range(images_per_layer):
            if image_index >= num_images:
                break

            angle = math.radians(i * angle_increment)  # Calculate angle in radians
            x = center[0] + int(radius * math.cos(angle))
            y = center[1] + int(radius * math.sin(angle))

            # Convert OpenCV image (NumPy array) to PIL Image
            image = Image.fromarray(cv2.cvtColor(cv2_images[image_index], cv2.COLOR_BGR2RGBA))

            # Rotate the image
            rotated_image = image.rotate(-math.degrees(angle) + 90, resample=Image.BICUBIC, expand=True)

            # Paste the rotated image onto the canvas
            collage.paste(rotated_image, (x - rotated_image.width // 2, y - rotated_image.height // 2), rotated_image)

            image_index += 1

        layer += 1

    # Convert the PIL Image collage back to an OpenCV image (NumPy array)
    collage = cv2.cvtColor(np.array(collage), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(output_path, collage)

# Example usage:
#from bigcrittercolor.helpers import _readBCCImgs
#from bigcrittercolor.helpers.image import _sortImgsByChannel

#cv2_images_list = _readBCCImgs(type="segment", data_folder="D:/bcc/ringtails")

#cv2_images_list = _sortImgsByChannel(cv2_images_list,colorspace="hls", channel=0,inverse=True)
#saveWhorlCollage(cv2_images_list,output_path="collage2.jpg")
