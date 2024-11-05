import os
import shutil

def _delDirsIfExists(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]  # Convert to list if a single directory is provided

    for dir_path in dirs:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)  # Remove directory and all its contents