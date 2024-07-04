import os
import shutil

def _createTempTestFolder(shared_temp_folder, temp_subfolder_name):
    # temp_subfolder is something like "empty" or "filtered" etc.
    temp_subfolder = os.path.join(shared_temp_folder, temp_subfolder_name)
    # new_temp_folder is always "tmp"
    new_temp_folder = shared_temp_folder + "/tmp"
    if os.path.exists(new_temp_folder):
        shutil.rmtree(new_temp_folder)
    shutil.copytree(temp_subfolder, new_temp_folder)

    return new_temp_folder