import os
import cv2

from bigcrittercolor.helpers.db import _writeDb, _createDb

def convertProjectToDb(data_folder, map_size_gb=50):
    map_size = map_size_gb * (1024 ** 3)
    # create the db
    _createDb(map_size=map_size,data_folder=data_folder)

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

    # # for each path and img name
    # for path, name in zip(img_paths,imgnames):
    #     # write to db
    #     img = cv2.imread(path)
    #     _writeDb(imgs=img,imgnames=name,rocksdb_path=data_folder + "/db")
    #     # rm
    #     os.remove(path)

    batch_size = 50

    # Determine the number of batches
    num_batches = (len(img_paths) + batch_size - 1) // batch_size

    for i in range(num_batches):
        # Calculate start and end indices for the current batch
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(img_paths))

        # Prepare lists for the current batch
        batch_paths = img_paths[start_idx:end_idx]
        batch_names = imgnames[start_idx:end_idx]
        batch_images = []

        # Read images and store them in a list
        for path in batch_paths:
            img = cv2.imread(path)
            batch_images.append(img)

        # Write the current batch to RocksDB
        _writeDb(imgs=batch_images, imgnames=batch_names, db_path=data_folder + "/db")

        # Remove the images processed in this batch
        for path in batch_paths:
            os.remove(path)

        if i % 20 == 0:
            print(i)


