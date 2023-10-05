from bigcrittercolor.clustextract import clusterMasksExtractSegs

def test_clusterextract_default():
    clusterMasksExtractSegs(filter_hw_ratio_minmax=(3, 100),filter_prop_img_minmax=(0, 0.25),
                       data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder")

def test_clusterextract_trimsides():
    clusterMasksExtractSegs(img_ids=None,filter_hw_ratio_minmax=(3, 100),filter_prop_img_minmax=(0, 0.25),
                       data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder")

def test_clusterextract_greysegs():
    clusterMasksExtractSegs(img_ids=None,sample_n=500, filter_prop_img_minmax=(0.001, 0.25),
                       data_folder="D:/dfly_appr_expr/appr1",cluster_raw_segs=True,cluster_n=7)
    #clusterMasksExtractSegs(img_ids=None, filter_hw_ratio_minmax=(3, 100), filter_prop_img_minmax=(0.001, 0.25),
    #                        data_folder="D:/dfly_appr_expr/appr1", cluster_raw_segs=True, cluster_n=7)


test_clusterextract_greysegs()