import cv2
import random

from bigcrittercolor.helpers import _readBCCImgs, _showImages, _getIDsInFolder, makeCollage

def showBCCImages(img_ids=None, sample_n=None, show_type="img", title="", data_folder=""):
    """ Show bigcrittercolor images, masks, and/or segments in a data folder
        Args:
            img_ids (list): the image IDs to draw images, masks, and/or segments for
            sample_n (int): the number of IDs to sample from img_ids
            show_type (str): whether to show images, masks, segments, or some combination of these
                Can be "img", "mask", "seg", "img_mask" (show image stitched with its mask for each ID),
                or "img_mask_seg" (show image, mask, and seg stitched together for each ID)
            data_folder (str): the path to the bigcrittercolor formatted data folder
    """

    # if no ids specified, get ids depending on what is asked for
    # for example if segments are asked for, keep only IDs that have segments, which will also have masks and images
    # conversely, if only images are asked for, keep all image IDs including those without segs or masks because we don't need those
    if img_ids is None:
        match show_type:
            case "img":
                img_ids = _getIDsInFolder(data_folder + "/all_images")
            case "mask":
                img_ids = _getIDsInFolder(data_folder + "/masks")
            case "seg":
                img_ids = _getIDsInFolder(data_folder + "/segments")
            case "img_mask":
                img_ids = _getIDsInFolder(data_folder + "/masks")
            case "img_mask_seg":
                img_ids = _getIDsInFolder(data_folder + "/segments")

    # if sample, sample from the IDs
    if sample_n is not None and len(img_ids) > sample_n:
        img_ids = random.sample(img_ids,sample_n)

    # read imgs, segs, or masks, or bind multiple together
    match show_type:
        case "img":
            imgs = _readBCCImgs(img_ids,type="img",data_folder=data_folder)
        case "mask":
            imgs = _readBCCImgs(img_ids,type="mask",data_folder=data_folder)
        case "seg":
            imgs = _readBCCImgs(img_ids,type="seg",data_folder=data_folder)
        case "img_mask":
            imgs = _readBCCImgs(img_ids, type="img", data_folder=data_folder)
            masks = _readBCCImgs(img_ids, type="mask", data_folder=data_folder)

            imgs = [makeCollage.makeCollage([img,mask],n_per_row=2) for img, mask in zip(imgs,masks)]

    # at the end we show what was kept
    _showImages(True, images=imgs, maintitle=title)