from bigcrittercolor import clusterColorsToPatterns

from bigcrittercolor.helpers import _getIDsInFolder
import random

#ids = ["WRK-WS-00165_fore","WRK-WS-00268_fore","WRK-WS-00396_fore"]
ids = _getIDsInFolder("D:/wing-color/data/segments")
ids = random.sample(ids, 1000)

clusterColorsToPatterns(img_ids=ids, blur_args={'type':"gaussian",'ksize':21}, equalize_args=None,
                        colorspace="cielab",
                        cluster_args={'n':5,'scale':True},
                        data_folder="D:/wing-color/data",
                        show_indv=False)
#clusterColorsToPatterns(img_ids=ids, cluster_args={'n':5},colorspace="cielab",blur_args={'type':"gaussian",'ksize':21},equalize_args=None,data_folder="D:/wing-color/data",
#                        show_indv=False)

#'find_n_minmax':(3,8)
#'merge_with_user_input':True