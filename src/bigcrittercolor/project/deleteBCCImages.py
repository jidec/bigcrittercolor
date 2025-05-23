import os
import lmdb

from bigcrittercolor.helpers.ids import _imgNameToID
from bigcrittercolor.helpers.db import _getDbIDs

def deleteBCCImages(type, ids, data_folder=''):
    use_db = os.path.exists(os.path.join(data_folder, "db"))
    deleted = []
    not_found = []

    if use_db:
        # Use LMDB
        db_path = os.path.join(data_folder, "db")
        with open(os.path.join(data_folder, "other", "map_size.txt"), 'r') as f:
            map_size = int(f.read().strip())
        env = lmdb.open(db_path, map_size=map_size)

        with env.begin(write=True) as txn:
            for img_id in ids:
                key = _idToImgName(img_id, type=type).encode('utf-8')
                if txn.get(key) is not None:
                    txn.delete(key)
                    deleted.append(key.decode('utf-8'))
                else:
                    not_found.append(key.decode('utf-8'))

        env.close()
    else:
        # Use file system
        folder_mapping = {
            "image": "all_images",
            "mask": "masks",
            "segment": "segments"
        }

        if type not in folder_mapping:
            raise ValueError(f"Unsupported type '{type}'. Use 'image', 'mask', or 'segment'.")

        target_dir = os.path.join(data_folder, folder_mapping[type])

        for img_id in ids:
            filename = _idToImgName(img_id, type=type)
            file_path = os.path.join(target_dir, filename)

            if os.path.exists(file_path):
                os.remove(file_path)
                deleted.append(file_path)
            else:
                not_found.append(file_path)

    print(f"Deleted {len(deleted)} items.")
    if not_found:
        print(f"{len(not_found)} items not found:")
        for f in not_found:
            print(f" - {f}")

def _idToImgName(img_id, type="image"):
    suffix = {
        "image": ".jpg",
        "mask": "_mask.png",
        "segment": "_segment.png"
    }.get(type, ".jpg")
    return img_id + suffix