import os
import lmdb

from bigcrittercolor.helpers.db import _createDb

def createBCCDataFolder(parent_folder, new_folder_name="bcc_data", use_db=False, map_size_gb=50):

    """ Create a data folder as required for a bigcrittercolor project

        Args:
            parent_folder (str): folder in which to place the new bigcrittercolor data folder
            new_folder_name (str): name of the new bigcrittercolor data folder
    """
    map_size = map_size_gb * (1024 ** 3)
    base_path = parent_folder + "/" + new_folder_name

    def mkdir_if_new(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    # make folder
    mkdir_if_new(base_path)

    # make images folders
    mkdir_if_new(base_path + "/all_images")
    mkdir_if_new(base_path + "/segments")
    mkdir_if_new(base_path + "/masks")
    mkdir_if_new(base_path + "/patterns")

    # make other folder
    mkdir_if_new(base_path + "/other")

    # make inat_download_records folder
    mkdir_if_new(base_path + "/other/inat_download_records")

    # make ml checkpoints
    mkdir_if_new(base_path + "/other/ml_checkpoints")

    # make raw records folder
    mkdir_if_new(base_path + "/other/raw_records")

    # make processing info folder
    mkdir_if_new(base_path + "/other/processing_info")
    with open(base_path + "/other/processing_info/failed_mask_infers.txt", 'w') as file:
        pass  # No need to write anything, just create the file

    # make dirs for new classifier training
    mkdir_if_new(base_path + "/other/filter_training/")
    mkdir_if_new(base_path + "/other/filter_training/test")
    mkdir_if_new(base_path + "/other/filter_training/test/good")
    mkdir_if_new(base_path + "/other/filter_training/test/bad")
    mkdir_if_new(base_path + "/other/filter_training/train")
    mkdir_if_new(base_path + "/other/filter_training/train/good")
    mkdir_if_new(base_path + "/other/filter_training/train/bad")

    if use_db:
        _createDb(data_folder=base_path,map_size=map_size)
