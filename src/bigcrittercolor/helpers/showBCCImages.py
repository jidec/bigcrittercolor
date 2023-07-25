import cv2
from bigcrittercolor.helpers import _showImages, _getIDsInFolder
import random

def showBCCImages(sample_n=10, img_ids=None, show_imgs=True, show_masks=False, show_segs=False, combine=False, data_folder=""):

    # if no ids specified, get ids depending on what is asked for
    # for example if segments are asked for, keep only IDs that have segments, which will also have masks and images
    # conversely, if only images are asked for, keep all image IDs including those without segs or masks because we don't need those
    if img_ids is None:
        if show_segs:
            img_ids = _getIDsInFolder(data_folder + "/segments")
        elif show_masks:
            img_ids = _getIDsInFolder(data_folder + "/masks")
        elif show_imgs:
            img_ids = _getIDsInFolder(data_folder + "/all_images")

    # if sample, sample from the IDs
    if sample_n is not None:
        img_ids = random.sample(img_ids,sample_n)

    if show_imgs:
        imgs = [cv2.imread(data_folder + "/all_images/" + id + ".jpg") for id in img_ids]
        _showImages(True, images=imgs, maintitle="Images")
    if show_masks:
        masks = [cv2.imread(data_folder + "/masks/" + id + "_masks.png") for id in img_ids]
        _showImages(True, images=masks, maintitle="Segments")
    if show_segs:
        segs = [cv2.imread(data_folder + "/segs/" + id + "_segment.png") for id in img_ids]
        _showImages(True, images=segs, maintitle="Segments")