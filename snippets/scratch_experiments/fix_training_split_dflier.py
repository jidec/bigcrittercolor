from bigcrittercolor.helpers.verticalize import _verticalizeImg2
import cv2
from bigcrittercolor.helpers import _getIDsInFolder
import numpy as np
import os

all_ids = _getIDsInFolder("D:/bcc/dfly_training/mask")
dfly_ids = _getIDsInFolder("D:/bcc/dfly_training/image")
dams_ids = [item for item in all_ids if item not in dfly_ids]

for id in dams_ids:
    # get damsel image
    #img = cv2.imread("D:/bcc/dfly_training/image/" + id + ".jpg")
    # mask
    #mask = cv2.imread("D:/bcc/unet_training/mask/" + id + "_mask.png")

    # delete mask
    os.remove("D:/bcc/dfly_training/mask/" + id + "_mask.png")

    # delete mask from
    #mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))