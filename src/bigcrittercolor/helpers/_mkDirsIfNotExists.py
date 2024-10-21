import os

def _mkDirsIfNotExists(dirs):
    if isinstance(dirs, str):
        dirs = [dirs]  # Convert to list if a single directory is provided

    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)