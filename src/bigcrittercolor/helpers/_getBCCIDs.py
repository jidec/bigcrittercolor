import os
import random
import rocksdbpy
import copy

from bigcrittercolor.helpers import _imgNameToID

def _getBCCIDs(type="image", sample_n=None, data_folder=''):
    use_db = os.path.exists(data_folder + "/db")

    if use_db:
        # Open the database with default options
        db = rocksdbpy.open_default(data_folder + "/db")
        # Create an iterator to go through the database
        iterator = db.iterator()
        # Initialize an empty list to store keys
        keys = []

        # Define suffix based on the type
        type_suffix = {
            "segment": "_segment",
            "mask": "_mask",
            "pattern": "_pattern"
        }.get(type)

        # Iterate over all key-value pairs in the database
        for key, value in iterator:
            # Decode the key
            decoded_key = key.decode('utf-8')
            # Filter keys based on type
            if type_suffix:
                if type_suffix in decoded_key:
                    keys.append(decoded_key)
            else:
                # For type "img", include keys that don't match any specific type
                if not any(suffix in decoded_key for suffix in ["_segment", "_mask", "_pattern"]):
                    keys.append(decoded_key)

        # Close the database to release resources
        #db.close()
        del iterator
        del db

        filenames = keys

    else:
        # Define a mapping of types to subdirectories
        folder_mapping = {
            "img": "all_images",
            "mask": "masks",
            "seg": "segments",
            "pattern": "patterns"
        }

        # Determine the correct subdirectory
        sub_dir = folder_mapping.get(type, "all_images")  # Default to "all_images"
        target_dir = os.path.join(data_folder, sub_dir)

        # List all files in the directory
        filenames = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]

    # for both approaches format to ID
    ids = [_imgNameToID(filename) for filename in filenames]

    # Optionally sample a subset of filenames
    if sample_n is not None and len(ids) > sample_n:
        ids = random.sample(ids, sample_n)

    return ids