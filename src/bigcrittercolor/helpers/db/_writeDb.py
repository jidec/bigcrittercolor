import cv2
import lmdb
import os

def _writeDb(imgs, imgnames, db_path):
    # Ensure input is in list format, even if only one image is provided
    if not isinstance(imgs, list):
        imgs = [imgs]
    if not isinstance(imgnames, list):
        imgnames = [imgnames]

    # Ensure the database path exists
    os.makedirs(db_path, exist_ok=True)

    # Read the map_size from the file
    with open(os.path.dirname(db_path) + "/other/map_size.txt", 'r') as f:
        map_size = int(f.read().strip())

    # Create or open the LMDB database at the specified path
    env = lmdb.open(db_path, map_size=map_size)  # Set a large map size; adjust as needed

    # Start a transaction in write mode
    with env.begin(write=True) as txn:
        # Iterate through each image and corresponding name
        for img, imgname in zip(imgs, imgnames):
            # Determine the correct encoding format based on the file extension
            if imgname.lower().endswith('.png'):
                encode_format = '.png'
            elif imgname.lower().endswith('.jpg') or imgname.lower().endswith('.jpeg'):
                encode_format = '.jpg'
            else:
                # Default to JPEG if no known extension is provided
                encode_format = '.jpg'

            try:
                # Encode the image to the appropriate format
                _, img_encoded = cv2.imencode(encode_format, img)
                img_bytes = img_encoded.tobytes()

                # Encode the image name to bytes to use as the key
                key_bytes = imgname.encode('utf-8')

                # Store the image bytes in the database using the key
                txn.put(key_bytes, img_bytes)

            except Exception as e:
                # Handle the error, e.g., log it or pass it up the chain
                print(f"Failed to encode or write the image {imgname} to the database...")

# import cv2
# import rocksdbpy
# def _writeDb(imgs, imgnames, db_path):
#     # Ensure input is in list format, even if only one image is provided
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     if not isinstance(imgnames, list):
#         imgnames = [imgnames]
#
#     # Open the RocksDB at the specified path
#     db = rocksdbpy.open_default(db_path)
#
#     # Iterate through each image and corresponding name
#     for img, imgname in zip(imgs, imgnames):
#         # Determine the correct encoding format based on the file extension
#         if imgname.lower().endswith('.png'):
#             encode_format = '.png'
#         elif imgname.lower().endswith('.jpg') or imgname.lower().endswith('.jpeg'):
#             encode_format = '.jpg'
#         else:
#             # Default to JPEG if no known extension is provided
#             encode_format = '.jpg'
#
#         # Encode the image to the appropriate format
#         _, img_encoded = cv2.imencode(encode_format, img)
#         img_bytes = img_encoded.tobytes()
#
#         # Encode the image name to bytes to use as the key
#         key_bytes = imgname.encode('utf-8')
#
#         # Store the image bytes in the database using the key
#         db.set(key_bytes, img_bytes)
#
#     # Close the database to ensure all operations are flushed and the database is safely closed
#     db.close()