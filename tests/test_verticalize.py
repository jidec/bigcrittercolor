from bigcrittercolor.helpers import _getIDsInFolder
from bigcrittercolor.helpers.imgtransforms import _verticalizeImg
import cv2

def test_verticalize():
    test_img = cv2.imread("D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/masks/INAT-18560075-1_mask.jpg")
    vert = _verticalizeImg(test_img,by_ellipse=False,crop_lr_size=50,show=True)
    cv2.imshow("Vert",vert)
    cv2.waitKey(0)

test_verticalize()

#"INAT-215236-1_mask.jpg"
#"INAT-6531057-2_mask.jpg"
#"INAT-166509820-1_mask.jpg"