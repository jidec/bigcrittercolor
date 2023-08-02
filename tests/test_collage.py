from bigcrittercolor.helpers import makeCollage, _getIDsInFolder
import cv2

def test_collage():
    ids = _getIDsInFolder("D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/all_images")
    # 2 ids
    ids = ids[0:2]
    imgs = [cv2.imread("D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/all_images/" + id + ".jpg") for id in ids]
    makeCollage(imgs, n_per_row=2, resize_wh=(100,100), show=True)

#test_collage()