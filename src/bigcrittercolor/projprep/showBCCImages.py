import cv2
import random

from bigcrittercolor.helpers import _readBCCImgs, _showImages, _getIDsInFolder, makeCollage

def showBCCImages(img_ids=None, sample_n=None, show_type="img", preclust_folder_name=None, title="", collage_resize_wh=(100,100), data_folder=""):
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
            case "mask" | "img_mask":
                img_ids = _getIDsInFolder(data_folder + "/masks")
            case "seg" | "img_mask_seg":
                img_ids = _getIDsInFolder(data_folder + "/segments")
            case "seg_preclust":
                img_ids = _getIDsInFolder(data_folder + "/patterns/" + preclust_folder_name)
            case "seg_pattern" | "seg_preclust_pattern" | "img_seg_pattern":
                img_ids = _getIDsInFolder(data_folder + "/patterns")

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

            imgs = [makeCollage([img,mask],resize_wh=collage_resize_wh,n_per_row=2) for img, mask in zip(imgs,masks)]
        case "img_seg_pattern":
            imgs = _readBCCImgs(img_ids, type="img", data_folder=data_folder)
            segs = _readBCCImgs(img_ids, type="seg", data_folder=data_folder)
            pats = _readBCCImgs(img_ids, type="pattern", data_folder=data_folder)
            imgs = [makeCollage([img, seg, pat], resize_wh=collage_resize_wh, n_per_row=3) for img, seg, pat in zip(imgs, segs, pats)]
        case "seg_pattern":
            segs = _readBCCImgs(img_ids, type="seg", data_folder=data_folder)
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

#showBCCImages(img_ids=["INAT-11256212-1"],show_type="seg_preclust_pattern",preclust_folder_name="phylo_preclustered",data_folder="D:/anac_tests")