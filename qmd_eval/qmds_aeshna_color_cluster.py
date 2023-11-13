from bigcrittercolor import clusterColorsToPatterns2
from bigcrittercolor.helpers import _getIDsInFolder
import random

ids = _getIDsInFolder("E:/aeshna_data/patterns/preclust")
ids = random.sample(ids,500)
clusterColorsToPatterns2(img_ids=ids,vert_resize=100,cluster_algo="kmeans",
                         data_folder="E:/aeshna_data",cluster_n=5, preclust_read_subfolder="preclust", preclustered=True,
                         group_cluster_raw_ids=True,show_indv=False,print_details=True)