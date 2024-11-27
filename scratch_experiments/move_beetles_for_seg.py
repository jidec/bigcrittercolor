from bigcrittercolor.helpers import _readBCCImgs
from bigcrittercolor.helpers.ids import _getIDsInFolder
import os
import cv2

ids = _getIDsInFolder("D:/bcc/beetle_appendage_segmenter/train/mask")

imgs = _readBCCImgs(img_ids=ids,type="raw_segment",data_folder="D:/bcc/beetles")

folder = "D:/bcc/beetle_appendage_segmenter/train/image"
# Save each image with its corresponding name
for image, name in zip(imgs, ids):
    # Create the full path with .jpg extension
    save_path = os.path.join(folder, f"{name}.jpg")

    # Save the image
    cv2.imwrite(save_path, image)