import lmdb

def _getDbIDs(data_folder, type):
    # Define the database path
    db_path = data_folder + "/db"

    # Read the map_size from the file
    with open(data_folder + "/other/map_size.txt", 'r') as f:
        map_size = int(f.read().strip())

    # Open the LMDB database at the specified path in read-only mode
    env = lmdb.open(db_path, map_size=map_size, readonly=True)

    # Initialize an empty list to store keys
    keys = []

    # Start a transaction in read mode
    with env.begin(write=False) as txn:
        # Create a cursor to iterate over the items in the database
        cursor = txn.cursor()

        # Use cursor.iternext(keys=True, values=False) to iterate over keys only
        for key in cursor.iternext(keys=True, values=False):
            # Decode the key and append to the list
            keys.append(key.decode('utf-8'))

    # Close the environment when done
    env.close()

    if type is "image":
        keys = [key for key in keys if not ("_segment" in key | "_mask" in key | "_pattern" in key)]
    else:
        # Define suffix based on the type
        type_suffix = {
            "segment": "_segment",
            "mask": "_mask",
            "pattern": "_pattern"
        }.get(type)

        keys = [key for key in keys if type_suffix in key]

    # Return the filtered filenames
    return keys