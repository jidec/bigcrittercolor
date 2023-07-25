from bigcrittercolor.helpers import _getIDsInFolder, _verthelpers
import cv2

def test_trimblobsidesonaxis():
    ids = _getIDsInFolder("D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/masks")
    for id in ids:
        mask = cv2.imread("D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder/masks/" + str(id) + "_mask.jpg",cv2.IMREAD_GRAYSCALE)
        _verthelpers._vertTrimSides(mask,show=True)
test_trimblobsidesonaxis()

#"INAT-215236-1_mask.jpg"
#"INAT-6531057-2_mask.jpg"
#"INAT-166509820-1_mask.jpg"