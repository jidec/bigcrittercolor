from bigcrittercolor.helpers.image import _imgToColorPatches, _blur, _reconstructImgFromPPD
import os
import cv2
from bigcrittercolor.helpers import _showImages

folder_path = "D:/bcc/ringtails/segments"

# Get a list of all files in the folder
files = os.listdir(folder_path)

for image_file in files:
    image_path = os.path.join(folder_path, image_file)

    # Load the image
    image = cv2.imread(image_path)

    # Apply the bilateral blur
    blurred = _blur(image, type="bilateral")

    # Convert the image to color patches
    ppd = _imgToColorPatches(blurred, id=1, show=False)

    # Reconstruct the image from color patches
    reconstruct = _reconstructImgFromPPD(ppd)

    # Show the reconstructed image
    _showImages(True, [image,blurred,reconstruct],save_folder="D:/bcc/ringtails/plots")