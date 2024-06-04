import cv2
import numpy as np
import random
import copy
import os

from bigcrittercolor.helpers import _getIDsInFolder
from bigcrittercolor.helpers import _readRocksdb
from bigcrittercolor.helpers.image import _segToMask, _format

def _readBCCImgs(img_ids=None, sample_n=None,type="image", preclust_folder_name=None, color_format=None, make_3channel=False, data_folder=''):
    # make sure that img_ids isn't modified
    img_ids = copy.deepcopy(img_ids)

    # if none load masks
    if img_ids is None:
        img_ids = _getIDsInFolder(data_folder + "/masks")

    # sample
    if sample_n is not None and len(img_ids) > sample_n:
        img_ids = random.sample(img_ids,sample_n)

    # if it's not a list, make it so
    was_single = False
    if not isinstance(img_ids, list):
        img_ids = [img_ids]
        was_single = True

    # if using rocksdb
    use_db = os.path.exists(data_folder + "/db")
    if use_db:
        match type:
            case "image":
                imgnames = [img_id + ".jpg" for img_id in img_ids]
                imgs = _readRocksdb._readRocksdb(imgnames=imgnames,rocksdb_path=data_folder+"/db")
            case "mask":
                imgnames = [img_id + "_mask.png" for img_id in img_ids]
                imgs = _readRocksdb._readRocksdb(imgnames=imgnames,rocksdb_path=data_folder+"/db")
            case "segment":
                imgnames = [img_id + "_segment.png" for img_id in img_ids]
                imgs = _readRocksdb._readRocksdb(imgnames=imgnames, rocksdb_path=data_folder + "/db")
            case "raw_segment":
                imgnames = [img_id + "_mask.png" for img_id in img_ids]
                imgs = _readRocksdb(imgnames=imgnames, rocksdb_path=data_folder + "/db")
                imgnames = [img_id + "_segment.png" for img_id in img_ids]
                masks = _readRocksdb(imgnames=imgnames, rocksdb_path=data_folder + "/db")
                imgs_masks = zip(imgs, masks)
                del imgs
                del masks
                segs = []
                for img, mask in imgs_masks:
                    if img is not None and mask is not None:
                        segs.append(
                            cv2.bitwise_and(img, img, mask=cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.uint8)))
                imgs = segs
            case "mask_from_segment":
                imgs = [cv2.imread(data_folder + "/segments/" + id + "_segment.png") for id in img_ids]
                imgs = [_segToMask(img) for img in imgs]
    # else using regular directory structure
    else:
        match type:
            case "image":
                imgs = [cv2.imread(data_folder + "/all_images/" + id + ".jpg") for id in img_ids]
            case "mask":
                imgs = [cv2.imread(data_folder + "/masks/" + id + "_mask.png") for id in img_ids]
            case "segment":
                imgs = [cv2.imread(data_folder + "/segments/" + id + "_segment.png") for id in img_ids]
            case "raw_segment":
                imgs = [cv2.imread(data_folder + "/all_images/" + id + ".jpg") for id in img_ids]
                masks = [cv2.imread(data_folder + "/masks/" + id + "_mask.png") for id in img_ids]
                imgs_masks = zip(imgs,masks)
                del imgs
                del masks
                segs = []
                i = 0
                for img, mask in imgs_masks:
                    #print(img_ids[i])
                    i = i + 1
                    if img is not None and mask is not None:
                        segs.append(cv2.bitwise_and(img, img, mask=cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.uint8)))
                imgs = segs
                #imgs = [cv2.bitwise_and(parent_img, parent_img,
                #                    mask=cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.uint8)) for
                #    mask, parent_img in imgs_masks]
            case "mask_from_segment":
                imgs = [cv2.imread(data_folder + "/segments/" + id + "_segment.png") for id in img_ids]
                imgs = [_segToMask(img) for img in imgs]
            case "pattern":
                if preclust_folder_name is not None:

                    print(data_folder + "/patterns/" + preclust_folder_name + "/" + img_ids[0] + "_pattern.png")
                    imgs = [cv2.imread(data_folder + "/patterns/" + preclust_folder_name + "/" + id + "_pattern.png") for id in img_ids]
                else:
                    imgs = [cv2.imread(data_folder + "/patterns/" + id + "_pattern.png") for id in img_ids]

    match color_format:
        case "grey":
            imgs = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in imgs]
        case "binary":
            imgs = [_segToMask(img) for img in imgs]

    if make_3channel:
        if len(imgs[0].shape) == 2:
            imgs = [_format(img, 'grey', 'grey3', False) for img in imgs]

    if was_single:
        imgs = imgs[0]

    return imgs

    #if img is not None:
    #    _showImages(show,[img],maintitle="Loaded Image")
    #else:
    #    _bprint(print, "Image is None at " + img_loc)