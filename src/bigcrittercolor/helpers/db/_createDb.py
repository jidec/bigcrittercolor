import lmdb

def _createDb(map_size,data_folder):
    env = lmdb.open(data_folder + "/db", map_size=map_size)

    # Save the map_size to a text file
    with open(data_folder + "/other/map_size.txt", 'w') as f:
        f.write(str(map_size))