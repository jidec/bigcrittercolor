from bigcrittercolor.helpers import _getIDsInFolder, _readBCCImgs
import random
import cv2

def extractRawSegs(img_ids=None, sample_n=None, data_folder=''):
    # if no ids specified load existing masks
    if img_ids is None:
        img_ids = _getIDsInFolder(data_folder + "/masks")
        if sample_n is not None:
            img_ids = random.sample(img_ids, sample_n)

    segs = _readBCCImgs(img_ids=img_ids, type="raw_segment", grey=False, data_folder=data_folder)

    segs_ids = [(x, y) for x, y in zip(segs, img_ids)]

    # write each segment naming it by its ID
    for seg_and_id in segs_ids:
        dest = data_folder + "/segments/" + seg_and_id[1] + "_segment.png"
        cv2.imwrite(dest, seg_and_id[0])

#extractRawSegs(data_folder="D:/dfly_appr_expr/appr1")