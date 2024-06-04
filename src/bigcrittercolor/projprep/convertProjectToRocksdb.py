import os
import cv2

from bigcrittercolor.helpers import _writeRocksdb

def convertProjectToRocksdb(data_folder):
    # list of image folders to convert images in
    img_folders_to_convert = ["all_images","masks","segments"]
    # paths to those folders
    folder_paths = [data_folder + "/" + img_folder for img_folder in img_folders_to_convert]
    img_paths = []
    imgnames = []

    # for each folder path
    for folder_path in folder_paths:
        for root, _, files in os.walk(folder_path):
            # for each file (image) in folder path
            for file in files:
                # Construct full file path
                file_path = os.path.join(root, file)
                img_paths.append(file_path)
                imgnames.append(file)

    # for each path and img name
    for path, name in zip(img_paths,imgnames):
        # write to db
        img = cv2.imread(path)
        _writeRocksdb(imgs=img,imgnames=name,data_folder=data_folder)
        # rm
        os.remove(file_path)


