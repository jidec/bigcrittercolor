from bigcrittercolor.helpers.verticalize import _verticalizeImg
import cv2
from bigcrittercolor.helpers import _getIDsInFolder

ids = _getIDsInFolder("D:/bcc/unet_training/image")

for id in ids:
    img = cv2.imread("D:/bcc/unet_training/image/" + id + ".jpg")
    mask = cv2.imread("D:/bcc/unet_training/mask/" + id + "_mask.png")

    line = _verticalizeImg(img, return_line=True)

    vert_mask = _verticalizeImg(mask, bound=False, flip=False, input_line=line)
    vert_img = _verticalizeImg(img, bound=False, flip=False, input_line=line)

    cv2.imwrite("D:/bcc/rot_norm_segmenter_img_strat/image/" + id + ".jpg", vert_img)
    cv2.imwrite("D:/bcc/rot_norm_segmenter_img_strat/mask/" + id + "_mask.png", vert_mask)
    #cv2.imshow("0",vert_mask)
    #cv2.imshow("1", vert_img)
    #cv2.waitKey(0)