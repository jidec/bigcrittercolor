from bigcrittercolor.helpers.verticalize import _verticalizeImg2
import cv2
from bigcrittercolor.helpers import _getIDsInFolder
import numpy as np

print("Starting")
#ids = _getIDsInFolder("D:/bcc/dfly_training/image")
ids = _getIDsInFolder("D:/bcc/damsels_segmenter/image")
ids = ids[134:]
for i, id in enumerate(ids):
    print(i)
    img = cv2.imread("D:/bcc/damsels_segmenter/image/" + id + ".jpg")
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    mask = cv2.imread("D:/bcc/damsels_segmenter/mask/" + id + "_mask.png")
    mask = cv2.resize(mask, (mask.shape[1] // 2, mask.shape[0] // 2))


    #cv2.imshow("0", mask)
    #cv2.waitKey(0)

    line = _verticalizeImg2(mask, lines_strategy="ellipse", best_line_metric="overlap_sym", return_line=True,show=False)

    #line = _verticalizeImg2(mask, lines_strategy="skeleton_hough", best_line_metric="overlap_sym", return_line=True,show=False)
    if isinstance(line, np.ndarray):
        continue
    vert_img, box, has_flipped = _verticalizeImg2(img, bound=True, flip=True, input_line=line, return_img_bb_flip=True)
    vert_mask = _verticalizeImg2(mask, input_line=line, bound=False, flip=False)
    x,y,w,h = box

    vert_mask = vert_mask[y:y + h, x:x + w]
    if has_flipped:
        vert_mask = cv2.flip(vert_mask, 0)

    cv2.imwrite("D:/bcc/damsels_segmenter_norm/image/" + id + ".jpg",vert_img)
    cv2.imwrite("D:/bcc/damsels_segmenter_norm/mask/" + id + "_mask.png", vert_mask)
    #cv2.imshow("0",vert_mask)
    #cv2.imshow("1", vert_img)
    #cv2.waitKey(0)