import cv2
from bigcrittercolor import inferMasks
from bigcrittercolor.helpers import _getIDsInFolder

ids = _getIDsInFolder("D:/bcc/dfly_appr_expr/appr_blur/og_imgs")

for id in ids:
    img = cv2.imread("D:/bcc/dfly_appr_expr/appr_blur/og_imgs/" + id + ".jpg")
    img = cv2.blur(img,(30, 30))

    cv2.imwrite("D:/bcc/dfly_appr_expr/appr_blur/all_images/" + id + ".jpg", img)


inferMasks(img_ids=None, skip_existing=False, data_folder="D:/bcc/dfly_appr_expr/appr_blur")