from bigcrittercolor import filterExtractSegs
from bigcrittercolor.helpers import _getIDsInFolder
#filterExtractSegs(img_ids=_getIDsInFolder("E:/aeshna_data/masks"), used_aux_segmodel=True,
#                   filter_hw_ratio_minmax=(3,100), filter_prop_img_minmax = (0.01, 0.9),
#                   data_folder="E:/aeshna_data")

filterExtractSegs(img_ids=_getIDsInFolder("E:/aeshna_data/masks"), used_aux_segmodel=True,
                   filter_hw_ratio_minmax=(3,100), cluster_params_dict={'pca_n':8}, filter_prop_img_minmax = (0.01, 0.9),
                   data_folder="E:/aeshna_data",feature_extractor="resnet18")
