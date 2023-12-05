from bigcrittercolor import clusterColorsToPatterns
from bigcrittercolor.helpers import _getIDsInFolder
import random

#ids = _getIDsInFolder("E:/aeshna_data/segments")
#ids = random.sample(ids,200)
clusterColorsToPatterns(data_folder="E:/aeshna_data",colorspace="cielab",
                        cluster_args={'n':11, 'algo':"kmeans", 'scale':True,'merge_with_user_input':True,'linkage':"complete"},
                        blur_args={'type':"bilateral",'d':9, 'sigma_color':150, 'sigma_space':150},show_indv=False)