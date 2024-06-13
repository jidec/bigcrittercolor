from bigcrittercolor.helpers import _getIDsInFolder, _showImages
from bigcrittercolor.helpers.verticalize import _verticalizeImg
import cv2
import numpy as np

ids = _getIDsInFolder("D:/bcc/dfly_appr_expr/appr1/masks")
ids = ids[700:]
#ids = ["INATRANDOM-56205494","INATRANDOM-5535855","INATRANDOM-55431557","INATRANDOM-55561130","INATRANDOM-62845118"]
for i, id in enumerate(ids):
    print(id)
    print(i)
    loc = "D:/bcc/dfly_appr_expr/appr1/masks/" + id + "_mask.png"
    img = cv2.imread(loc)#, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    src_img = cv2.imread("D:/bcc/dfly_appr_expr/appr1/all_images/" + id + ".jpg")
    src_img = cv2.resize(src_img, (src_img.shape[1] // 2, src_img.shape[0] // 2))

    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Dilate the image to expand the white parts
    #img = cv2.dilate(img, kernel, iterations=3

    img1 = _verticalizeImg(img,lines_strategy="skeleton_hough", best_line_metric="overlap_sym", sh_rho=3, sh_theta=np.pi/30, sh_thresh=50, show=False)
    img2 = _verticalizeImg(img,lines_strategy="skeleton_hough", best_line_metric="overlap_sym", sh_rho=3, sh_theta=np.pi/30, sh_thresh=25, show=False)
    img3 = _verticalizeImg(img,lines_strategy="skeleton_hough", best_line_metric="skinniness", show=False)


    #pix_sym_img = _verticalizeImg2(img,lines_strategy="skeleton_hough", best_line_metric="npix_sym",show=False)
    #skinny_img = _verticalizeImg2(img,lines_strategy="skeleton_hough", best_line_metric="skinniness",show=False)
    #dark_img = _verticalizeImg2(img,src_img=src_img,lines_strategy="skeleton_hough",best_line_metric="avg_darkness",show=False)
    #_showImages(True, [img, src_img, overlap_sym_img, pix_sym_img, skinny_img, dark_img], ["Mask", "Source", "Overlap Vert","Pix Sym Vert", "Skinny","Dark"])
    _showImages(True, [img, src_img, img1, img2,img3], [id, "Source", "1", "2", "3"])
