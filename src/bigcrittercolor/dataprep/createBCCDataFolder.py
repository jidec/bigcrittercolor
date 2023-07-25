import os

def createBCCDataFolder(parent_folder, new_folder_name="bcc_data"):

    """
    Create a data folder as required for a bigcrittercolor project

    :param str parent_folder: folder in which to place the new bigcrittercolor data folder
    :param str new_folder_name: name of the new bigcrittercolor data folder

    """

    base_path = parent_folder + "/" + new_folder_name

    # make folder
    os.mkdir(base_path)

    # make images folders
    os.mkdir(base_path + "/all_images")
    os.mkdir(base_path + "/segments")
    os.mkdir(base_path + "/masks")
    os.mkdir(base_path + "/patterns")

    # make other folder
    os.mkdir(base_path + "/other")

    # make inat_download_records folder
    os.mkdir(base_path + "/other/inat_download_records")

    # make ml checkpoints
    os.mkdir(base_path + "/other/ml_checkpoints")

    # make raw records folder
    os.mkdir(base_path + "/other/raw_records")

    # make dirs for new classifier training
    os.mkdir(base_path + "/other/filter_training/")
    os.mkdir(base_path + "/other/filter_training/test")
    os.mkdir(base_path + "/other/filter_training/test/good")
    os.mkdir(base_path + "/other/filter_training/test/bad")
    os.mkdir(base_path + "/other/filter_training/train")
    os.mkdir(base_path + "/other/filter_training/train/good")
    os.mkdir(base_path + "/other/filter_training/train/bad")