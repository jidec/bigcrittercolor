import os
import lmdb

from bigcrittercolor.helpers.db import _createDb
from bigcrittercolor.helpers import _bprint

def createBigcrittercolorDataFolder(parent_folder, new_folder_name="bcc_data", use_db=False, map_size_gb=50, print_steps=True):

    """ Create a data folder as required for a bigcrittercolor project

        Args:
            parent_folder (str): folder in which to place the new bigcrittercolor data folder
            new_folder_name (str): name of the new bigcrittercolor data folder
            use_db (bool): whether to use an LMDB database to store images (primarily for alleviating storage issues on some cluster architectures)
            map_size_gb (int): maximum size of the LMDB store in GB
    """

    map_size = map_size_gb * (1024 ** 3)
    base_path = parent_folder + "/" + new_folder_name

    def mkdir_if_new(path):
        if not os.path.isdir(path):
            os.mkdir(path)

    def mkdirs(base_path,folder_strs):
        for str in folder_strs:
            mkdir_if_new(base_path + str)

    # make folder
    mkdir_if_new(base_path)

    # make images folders
    mkdirs(base_path,["/all_images","/segments","/masks","/patterns","/plots"])

    # make other folder
    mkdir_if_new(base_path + "/other")

    # make other subfolders
    mkdirs(base_path + "/other", ["/bioencoder", "/bioencoder/bioencoder_encodings_images","/bioencoder/bioencoder_training",
                                  "/download", "/download/split_download_records","/download/raw_image_downloads",
                                  "/manual_coding_datasets",
                                  "/ml_checkpoints",
                                  "/my_gbif_darwincore",
                                  "/species_genus_exemplars"])

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

    _bprint(print_steps, "Created folder at " + base_path + ".")
    #_bprint(print_steps, "If downloading images using DarwinCore records downloaded from GBIF (recommended when analyzing large clades) don't forget to unzip the archive to " + base_path + "/other/my_gbif_darwincore")

#createBCCDataFolder(parent_folder="D:/bcc",new_folder_name="all_beetles_download_obsorg")
