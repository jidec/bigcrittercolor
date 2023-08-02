import cv2
from bigcrittercolor.helpers import _bprint,_showImages

def _readBCCImgs(img_ids, type="img", grey=False, print=False, show=False, data_folder=''):
    if not isinstance(img_ids, list):
        img_ids = [img_ids]
    match type:
        case "img":
            imgs = [cv2.imread(data_folder + "/all_images/" + id + ".jpg") for id in img_ids]
        case "mask":
            imgs = [cv2.imread(data_folder + "/masks/" + id + "_mask.png") for id in img_ids]
        case "seg":
            imgs = [cv2.imread(data_folder + "/segments/" + id + "_segment.png") for id in img_ids]

    if grey:
        imgs = [cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) for img in imgs]

    return imgs

    #if img is not None:
    #    _showImages(show,[img],maintitle="Loaded Image")
    #else:
    #    _bprint(print, "Image is None at " + img_loc)