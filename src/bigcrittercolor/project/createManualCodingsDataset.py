import pandas as pd
import random
import os
import cv2

from bigcrittercolor.helpers import _getBCCIDs, _bprint, _readBCCImgs
from bigcrittercolor.helpers.ids import _getSpeciesBalancedIDs

# we have 3 of the filtered segments from each species
def createManualCodingsDataset(dataset_name="example", type="segment", n_per_species=2, print_steps=True, print_details=True,data_folder=''):

    # exclude existing ids in codings folders...

    # we get balanced ids from segments
    balanced_ids = _getSpeciesBalancedIDs(type="segment",n_per_species=n_per_species,
                                          print_steps=print_steps,print_details=print_details,
                                          data_folder=data_folder)

    imgs = _readBCCImgs(img_ids=balanced_ids,type=type,data_folder=data_folder)

    # create dataset folder
    dataset_folder = data_folder + "/other/manual_coding_datasets/" + dataset_name
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    # Save each image to the folder with the corresponding ID as the filename
    for img, img_id in zip(imgs, balanced_ids):
        # Create a file name for each image using its ID
        file_path = os.path.join(dataset_folder, f"{img_id}.png")
        # Save the image using cv2
        cv2.imwrite(file_path, img)

#createManualCodingsDataset(type="segment",data_folder="D:/bcc/ringtails")