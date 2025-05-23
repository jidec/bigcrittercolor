import numpy as np
import cv2
from PIL import Image, ImageOps
import math
from bigcrittercolor.helpers.image import _resizeImgsToAvgSize

def saveSpiralWhorlCollage(cv2_images, output_path, canvas_size=(5000, 5000), center=(2500, 2500), spacing=50, angle_increment=15):
    """
    Create a spiral whorl collage from a list of OpenCV images.

    Args:
        cv2_images (list): List of OpenCV images (NumPy arrays).
        output_path (str): Path to save the resulting collage.
        canvas_size (tuple): Size of the output canvas (width, height).
        center (tuple): Center of the spiral (x, y).
        spacing (int): Distance between consecutive images in the spiral.
        angle_increment (float): Angle increment in degrees between consecutive images.

    Returns:
        None: Saves the resulting collage to the specified output path.
    """
    cv2_images = _resizeImgsToAvgSize(cv2_images)

    # Create a blank canvas (RGBA for transparency)
    collage = Image.new("RGBA", canvas_size, (255, 255, 255, 0))

    angle = 0  # Start angle
    radius = 0  # Start radius

    for image in cv2_images:
        # Calculate position based on spiral equation
        x = center[0] + int(radius * math.cos(math.radians(angle)))
        y = center[1] + int(radius * math.sin(math.radians(angle)))

        # Convert OpenCV image (NumPy array) to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGBA))

        # Rotate the image to align with the spiral angle
        rotated_image = pil_image.rotate(-angle, resample=Image.BICUBIC, expand=True)

        # Paste the rotated image onto the canvas
        collage.paste(rotated_image, (x - rotated_image.width // 2, y - rotated_image.height // 2), rotated_image)

        # Update angle and radius for the next image
        angle += angle_increment
        radius += spacing / (2 * math.pi)  # Increase radius proportionally to maintain spacing

    # Convert the PIL Image collage back to an OpenCV image (NumPy array)
    collage = cv2.cvtColor(np.array(collage), cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(output_path, collage)

# Example usage:
#from bigcrittercolor.helpers import _readBCCImgs
#from bigcrittercolor.helpers.image import _sortImgsByChannel

#cv2_images_list = _readBCCImgs(type="segment", data_folder="D:/bcc/ringtails")

#cv2_images_list = _sortImgsByChannel(cv2_images_list,colorspace="hls", channel=0,inverse=True)
#saveSpiralWhorlCollage(cv2_images_list,output_path="collage3.jpg")
