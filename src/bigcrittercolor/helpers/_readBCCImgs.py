import cv2
import numpy as np
from bigcrittercolor.helpers import _getIDsInFolder
from bigcrittercolor.helpers.image import _segToMask, _format

def _readBCCImgs(img_ids=None, type="img", color_format=None, make_3channel=False, print_steps=False, show=False, data_folder=''):
    if img_ids is None:
        img_ids = _getIDsInFolder(data_folder + "/masks")
    was_single =False
    if not isinstance(img_ids, list):
        img_ids = [img_ids]
        was_single = True
    match type:
        case "img":
            imgs = [cv2.imread(data_folder + "/all_images/" + id + ".jpg") for id in img_ids]
        case "mask":
            imgs = [cv2.imread(data_folder + "/masks/" + id + "_mask.png") for id in img_ids]
        case "seg":
            imgs = [cv2.imread(data_folder + "/segments/" + id + "_segment.png") for id in img_ids]
        case "raw_seg":
            imgs = [cv2.imread(data_folder + "/all_images/" + id + ".jpg") for id in img_ids]
            masks = [cv2.imread(data_folder + "/masks/" + id + "_mask.png") for id in img_ids]
            imgs_masks = zip(imgs,masks)
            del imgs
            del masks
            segs = []
            i = 0
            for img, mask in imgs_masks:
                print(img_ids[i])
                i = i + 1
                #if img is not None and mask is not None:
                segs.append(cv2.bitwise_and(img, img, mask=cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.uint8)))
            imgs = segs
            #imgs = [cv2.bitwise_and(parent_img, parent_img,
            #                    mask=cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY).astype(np.uint8)) for
            #    mask, parent_img in imgs_masks]
        case "mask_from_seg":
            imgs = [cv2.imread(data_folder + "/segments/" + id + "_segment.png") for id in img_ids]
            imgs = [_segToMask(img) for img in imgs]

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