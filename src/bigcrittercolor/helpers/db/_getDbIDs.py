import lmdb

def _getDbIDs(data_folder, type):
    # Define the database path
    db_path = data_folder + "/db"

    # Read the map_size from the file
    with open(data_folder + "/other/map_size.txt", 'r') as f:
        map_size = int(f.read().strip())

    # Open the LMDB database at the specified path in read-only mode
    env = lmdb.open(db_path, map_size=map_size, readonly=True)

    # Define suffix based on the type
    type_suffix = {
        "segment": "_segment",
        "mask": "_mask",
        "pattern": "_pattern"
    }.get(type)

    # Initialize an empty list to store keys
    keys = []

    # Start a transaction in read mode
    with env.begin() as txn:
        # Create a cursor to iterate over the items in the database
        cursor = txn.cursor()

        # Iterate over all key-value pairs in the database
        for key, _ in cursor:
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

    # Close the environment when done
    env.close()

    # Return the filtered filenames
    return keys