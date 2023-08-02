from bigcrittercolor.maskfilter import clusterMasksExtractSegs

def test_filterclustermasks():
    clusterMasksExtractSegs(img_ids=None,filter_hw_ratio_minmax=(3, 100),filter_prop_img_minmax=(0, 0.25),
                       data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder")
test_filterclustermasks()