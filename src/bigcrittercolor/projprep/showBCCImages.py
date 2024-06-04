import cv2
import random
import os

from bigcrittercolor.helpers import _readBCCImgs, _showImages, _getIDsInFolder, makeCollage, _getBCCIDs

def showBCCImages(img_ids=None, sample_n=None, type="image", preclust_folder_name=None, title="", collage_resize_wh=(100,100), data_folder=""):
    """ Show bigcrittercolor images, masks, and/or segments in a data folder
        Args:
            img_ids (list): the image IDs to draw images, masks, and/or segments for
            sample_n (int): the number of IDs to sample from img_ids
            type (str): whether to show images, masks, segments, or some combination of these
                Can be "img", "mask", "seg", "img_mask" (show image stitched with its mask for each ID),
                or "img_mask_seg" (show image, mask, and seg stitched together for each ID)
            data_folder (str): the path to the bigcrittercolor formatted data folder
    """

    # if no ids specified, get ids depending on what is asked for
    # for example if segments are asked for, keep only IDs that have segments, which will also have masks and images
    # conversely, if only images are asked for, keep all image IDs including those without segs or masks because we don't need those
    if img_ids is None:
        match type:
            case "image":
                img_ids = _getBCCIDs(type="image",data_folder=data_folder)
            case "mask" | "img_mask":
                img_ids = _getBCCIDs(type="mask",data_folder=data_folder)
            case "segment" | "image_mask_segment":
                img_ids = _getBCCIDs(type="segment",data_folder=data_folder)
            case "segment_preclust":
                img_ids = _getIDsInFolder(data_folder + "/patterns/" + preclust_folder_name)
            case "segment_pattern" | "segment_preclust_pattern" | "image_segment_pattern":
                img_ids = _getIDsInFolder(data_folder + "/patterns")

    # if sample, sample from the IDs
    if sample_n is not None and len(img_ids) > sample_n:
        img_ids = random.sample(img_ids,sample_n)

    # read imgs, segs, or masks, or bind multiple together
    match type:
        case "image":
            imgs = _readBCCImgs(img_ids,type="image",data_folder=data_folder)
        case "mask":
            imgs = _readBCCImgs(img_ids,type="mask",data_folder=data_folder)
        case "segment":
            imgs = _readBCCImgs(img_ids,type="segment",data_folder=data_folder)
        case "image_mask":
            imgs = _readBCCImgs(img_ids, type="image", data_folder=data_folder)
            masks = _readBCCImgs(img_ids, type="mask", data_folder=data_folder)
            imgs = [makeCollage([img,mask],resize_wh=collage_resize_wh,n_per_row=2) for img, mask in zip(imgs,masks)]
        case "img_seg_pattern":
            imgs = _readBCCImgs(img_ids, type="image", data_folder=data_folder)
            segs = _readBCCImgs(img_ids, type="segment", data_folder=data_folder)
            pats = _readBCCImgs(img_ids, type="pattern", data_folder=data_folder)
            imgs = [makeCollage([img, seg, pat], resize_wh=collage_resize_wh, n_per_row=3) for img, seg, pat in zip(imgs, segs, pats)]
        case "seg_pattern":
            segs = _readBCCImgs(img_ids, type="segment", data_folder=data_folder)
            pats = _readBCCImgs(img_ids, type="pattern", data_folder=data_folder)
            imgs = [makeCollage([seg, pat], resize_wh=collage_resize_wh, n_per_row=2) for seg, pat in zip(segs, pats)]
        case "seg_preclust":
            segs = _readBCCImgs(img_ids, type="seg", data_folder=data_folder)
            preclusts = _readBCCImgs(img_ids, type="pattern", preclust_folder_name=preclust_folder_name,
                                     data_folder=data_folder)
            imgs = [makeCollage([seg, preclust], resize_wh=collage_resize_wh, n_per_row=2) for seg, preclust in
                    zip(segs, preclusts)]
        case "seg_preclust_pattern":
            segs = _readBCCImgs(img_ids, type="seg", data_folder=data_folder)
            preclusts = _readBCCImgs(img_ids, type="pattern", preclust_folder_name = preclust_folder_name,data_folder=data_folder)
            pats = _readBCCImgs(img_ids, type="pattern", data_folder=data_folder)
            imgs = [makeCollage([seg, preclust, pat], resize_wh=collage_resize_wh, n_per_row=3) for seg, preclust, pat in zip(segs, preclusts, pats)]
    # at the end we show what was kept
    _showImages(True, images=imgs, maintitle=title)