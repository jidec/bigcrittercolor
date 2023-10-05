from bigcrittercolor.helpers.imgtransforms import _getRotInvariantBlobSymmetry, _passesRotInvariantSym
import cv2
import numpy as np
from bigcrittercolor.helpers import _readBCCImgs, _getIDsInFolder, _showImages
import random

def test_getSymmetry():
    #imgnames = ["INAT-84092769-2_mask.jpg", "INAT-215236-2_mask.jpg", "INAT-215236-1_mask.png",
    #            "INAT-6531057-2_mask.png", "INAT-169500716-1_mask.png"]
    imgnames2 = ["INATRANDOM-746982_mask.png", "INATRANDOM-350136_mask.png",
                 "INATRANDOM-4033807_mask.png", "INATRANDOM-4313113_mask.png"]
    #for name in imgnames:
    #    img = cv2.imread("D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/masks/" + name)
    #    greyu8 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.uint8)
    #    print(_getRotInvariantBlobSymmetry(greyu8,show=True))

    for name in imgnames2:
        img = cv2.imread("D:/dfly_appr_expr/appr1/masks/" + name)
        greyu8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        print(_getRotInvariantBlobSymmetry(greyu8, show=True))

#test_getSymmetry()

def test_sym_filter():
    ids = random.sample(_getIDsInFolder("D:/dfly_appr_expr/appr1/masks"),200)
    imgs = _readBCCImgs(img_ids=ids,type="mask",data_folder="D:/dfly_appr_expr/appr1")
    print("done with load")
    imgs = [img for img in imgs if _passesRotInvariantSym(img, 0.4,show=False)]
    print("done with sym")
    _showImages(True,imgs)

test_sym_filter()