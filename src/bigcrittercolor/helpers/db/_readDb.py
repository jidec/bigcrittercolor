import cv2
import lmdb
import numpy as np
import os

def _readDb(imgnames, db_path):
    # Ensure imgnames is a list, even if only one name is provided
    if not isinstance(imgnames, list):
        imgnames = [imgnames]

    # Read the map_size from the file
    with open(os.path.dirname(db_path) + "/other/map_size.txt", 'r') as f:
        map_size = int(f.read().strip())

    # open db
    env = lmdb.open(db_path, map_size=map_size, readonly=True)

    images = []
    with env.begin() as txn:
        for imgname in imgnames:
            # Encode the image name to bytes to use as the key
            key_bytes = imgname.encode('utf-8')

            # Retrieve the image bytes from the database using the key
            img_bytes = txn.get(key_bytes)
            if img_bytes:
                # Convert the bytes back to a numpy array and decode
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
                images.append(img)
            else:
                # Append None if no image was found for the given key
                images.append(None)

    #env.close()
    return images

# import cv2
# import rocksdbpy
# import numpy as np
#
# # can take a list or individual imgnames, ALWAYS returns a list
# def _readDb(imgnames, db_path, opened_db=None):
#     # Ensure imgnames is a list, even if only one name is provided
#     if not isinstance(imgnames, list):
#         imgnames = [imgnames]
#
#     #db = rocksdbpy.open_default(db_path)
#     db = rocksdbpy.open_for_readonly(db_path)
#
#     images = []
#     for imgname in imgnames:
#         # Encode the image name to bytes to use as the key
#         key_bytes = imgname.encode('utf-8')
#
#         # Retrieve the image bytes from the database using the key
#         img_bytes = db.get(key_bytes)
#
#         # Convert the bytes back to a numpy array and decode if found
#         nparr = np.frombuffer(img_bytes, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
#         images.append(img)
#
#         #if img_bytes:
#         #else:
#         #    # Append None if no image was found for the given key
#         #    images.append(None)
#
#     # Close the database
#     if not opened_db:
#         db.close()
#     #del db
#
#     return images
#     # Return the list of images or a single image if only one was requested
#     #return images if len(images) > 1 else images[0]