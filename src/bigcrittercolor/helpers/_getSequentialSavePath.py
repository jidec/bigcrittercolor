import os

# given a base path like /other/bioencoder/inferred_encodings/bioencodings.csv, add -1,-2,-3 etc when necessary
def _getSequentialSavePath(base_path):
    # Extract the directory and filename with extension from the base path
    dir_path, filename = os.path.split(base_path)

    # Separate filename and extension to avoid issues with periods in the path
    if '.' in filename:
        base_name, extension = filename.rsplit('.', 1)
    else:
        base_name, extension = filename, ""

    # Check the directory for files with a similar base name and different suffixes
    existing_files = [f for f in os.listdir(dir_path) if f.startswith(base_name)]

    # Find the highest suffix number among the files
    max_index = 0
    for file in existing_files:
        try:
            # Look for the suffix in the format '_1.csv', '_2.csv', etc.
            index = int(file.split("_")[-1].split(".")[0])
            if index > max_index:
                max_index = index
        except ValueError:
            continue

    # Increment max_index to get the new suffix
    new_index = max_index + 1
    # Return the full path with the incremented suffix, re-attaching the extension
    new_path = os.path.join(dir_path,
                            f"{base_name}_{new_index}.{extension}" if extension else f"{base_name}_{new_index}")
    return new_path