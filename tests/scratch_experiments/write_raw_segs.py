import cv2

from bigcrittercolor.helpers import _readBCCImgs
from bigcrittercolor.helpers import _getIDsInFolder

imgs = []
for id in _getIDsInFolder("D:/dfly_appr_expr/appr1/masks"):
    print(id)
    img = _readBCCImgs(img_ids=[id], type="raw_seg",data_folder="D:/dfly_appr_expr/appr1")[0]
    imgs.append(img)