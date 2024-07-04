from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, filterExtractSegs, clusterColorsToPatterns
from bigcrittercolor.helpers import _getBCCIDs
# illum inliers
#

ids = _getBCCIDs(type="mask",records_filter_dict={'species':"Erpetogomphus designatus"},data_folder="D:/bcc/ringtails")
# 4 or 5 could be better with inception
filterExtractSegs(img_ids=ids, color_format_to_cluster = "grey", used_aux_segmodel=True,
    filter_hw_ratio_minmax = None, filter_prop_img_minmax = None, filter_symmetry_min = None, filter_not_intersects_sides=False, # filters
    illum_outliers_percent=0.2,feature_extractor="resnet18",data_folder="D:/bcc/ringtails")

