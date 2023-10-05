from bigcrittercolor.helpers import _getIDsInFolder, _readBCCImgs
from bigcrittercolor.helpers.imgtransforms import _verticalizeImg
import cv2

def test_verticalize():
    test_img = cv2.imread("D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/masks/INAT-18560075-1_mask.jpg")
    vert = _verticalizeImg(test_img,by_ellipse=False,trim_sides_on_axis_size=50,show=True)
    cv2.imshow("Vert",vert)
    cv2.waitKey(0)

def test_verticalize_many():
    ids = _getIDsInFolder("D:/dfly_appr_expr/appr2/masks")
    imgs = _readBCCImgs(img_ids=ids,type="mask",
                        data_folder="D:/dfly_appr_expr/appr2")

    imgs = [cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))) for img in imgs]

    for id, img in zip(ids,imgs):
        print(id)
        vert = _verticalizeImg(img,show=True)

#test_verticalize_many()

#"INAT-215236-1_mask.jpg"
#"INAT-6531057-2_mask.jpg"
#"INAT-166509820-1_mask.jpg"

def test_verticalize_on_axis():
    test_img = cv2.imread("D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/masks/INAT-18560075-1_mask.jpg")
    vert = _verticalizeImg(test_img,by_ellipse=False,trim_sides_on_axis_size=50,show=True)
    cv2.imshow("Vert",vert)
    cv2.waitKey(0)

def test_verticalize_many_on_axis_show():
    ids = _getIDsInFolder("D:/dfly_appr_expr/appr2/masks")
    imgs = _readBCCImgs(img_ids=ids, type="mask",
                        data_folder="D:/dfly_appr_expr/appr2")

    imgs = [cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))) for img in imgs]

    for id, img in zip(ids, imgs):
        print(id)
        vert = _verticalizeImg(img, by_ellipse=False, trim_sides_on_axis_size=50, show=True)

def test_verticalize_many_by_symmetry():
    ids = _getIDsInFolder("D:/dfly_appr_expr/appr2/masks")
    imgs = _readBCCImgs(img_ids=ids, type="mask",
                        data_folder="D:/dfly_appr_expr/appr2")

    imgs = [cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))) for img in imgs]

    for id, img in zip(ids, imgs):
        print(id)
        vert = _verticalizeImg(img, by_ellipse=False, trim_sides_on_axis_size=50, show=True)

def test_vert_polygon():
    ids = _getIDsInFolder("D:/dfly_appr_expr/appr2/masks")
    imgs = _readBCCImgs(img_ids=ids, type="mask",
                        data_folder="D:/dfly_appr_expr/appr2")

    imgs = [cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))) for img in imgs]

    for id, img in zip(ids, imgs):
        print(id)
        vert = _verticalizeImg(img, strategy="polygon", polygon_e_mult=0.01, show=True)

def test_vert_symifsym():
    ids = _getIDsInFolder("D:/dfly_appr_expr/appr2/masks")
    imgs = _readBCCImgs(img_ids=ids, type="mask",
                        data_folder="D:/dfly_appr_expr/appr2")

    imgs = [cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))) for img in imgs]

    for id, img in zip(ids, imgs):
        print(id)
        vert = _verticalizeImg(img, strategy="polygon",sym_if_sym=0.4,polygon_e_mult=0.01, show=True)

test_vert_symifsym()