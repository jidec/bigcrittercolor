from bigcrittercolor.maskfilter import filterClusterMasksExtractSegs

def test_filterclustermasks():
    filterClusterMasksExtractSegs(img_ids=None,filter_hw_ratio_minmax=(3, 100),filter_prop_img_minmax=(0, 0.25),
                       data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder")
test_filterclustermasks()