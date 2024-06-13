from bigcrittercolor.project import createBCCDataFolder
from bigcrittercolor.helpers import _getIDsInFolder
import shutil
from bigcrittercolor.segment import inferMasks
import cv2
from bigcrittercolor.helpers import _readBCCImgs

#createBCCDataFolder(parent_folder="D:/GitProjects/bigcrittercolor/tests", new_folder_name= "train_import")
#ids1 = _getIDsInFolder("E:/dragonfly-patterner/data/other/training_dirs/segmenter/test/mask")
#ids2 = _getIDsInFolder("E:/dragonfly-patterner/data/other/training_dirs/segmenter/train/mask")
#ids = ids1 + ids2

#for id in ids:
#    src = "E:/dragonfly-patterner/data/all_images/" + id + ".jpg"
#    dst = "D:/GitProjects/bigcrittercolor/tests/train_import/all_images/" + id + ".jpg"
#    shutil.copy(src, dst)

#inferMasks(img_ids=None,skip_existing=True,data_folder="D:/GitProjects/bigcrittercolor/tests/train_import",
#               text_prompt="animal", strategy="prompt1", erode_kernel_size=3, show=True, show_indv=True)
ids = _getIDsInFolder("D:/GitProjects/bigcrittercolor/tests/train_import/masks")
n_test=70
for index, id in enumerate(ids):
    # load raw seg
    seg = _readBCCImgs(img_ids=id,type="raw_seg",data_folder="D:/GitProjects/bigcrittercolor/tests/train_import")
    # load mask
    mask = cv2.imread("D:/GitProjects/bigcrittercolor/tests/train_import/prev_training_masks/" + id + "_mask.jpg")

    # get
    if index < n_test:
        # write image seg to test/image
        cv2.imwrite("D:/GitProjects/bigcrittercolor/tests/full_segmenter/test/image/" + id + ".jpg", seg)
        # write mask to test/mask
        cv2.imwrite("D:/GitProjects/bigcrittercolor/tests/full_segmenter/test/mask/" + id + "_mask.png", mask)
    else:
        # write image seg to train/image
        cv2.imwrite("D:/GitProjects/bigcrittercolor/tests/full_segmenter/train/image/" + id + ".jpg", seg)
        # write mask to train/mask
        cv2.imwrite("D:/GitProjects/bigcrittercolor/tests/full_segmenter/train/mask/" + id + "_mask.png", mask)


