import os
import cv2
from bigcrittercolor.helpers import _writeRocksdb

def _writeBCCImgs(imgs, imgnames, data_folder=''):
    use_db = os.path.exists(data_folder + "/db")

    if use_db:
        _writeRocksdb(imgs, imgnames, rocksdb_path=data_folder + "/db")
    else:
        # Ensure input is in list format, even if only one image is provided
        if not isinstance(imgs, list):
            imgs = [imgs]
        if not isinstance(imgnames, list):
            imgnames = [imgnames]

        # Iterate through each image and corresponding name
        for img, imgname in zip(imgs, imgnames):
            # Determine the target subdirectory
            if '_segment' in imgname:
                target_folder = 'segments'
            elif '_mask' in imgname:
                target_folder = 'masks'
            elif '_pattern' in imgname:
                target_folder = 'patterns'
            else:
                target_folder = 'all_images'

            target_path = os.path.join(data_folder, target_folder)
            # Define the path for the image file
            file_path = os.path.join(target_path, imgname)

            # Write the image file to the appropriate directory
            cv2.imwrite(file_path, img)