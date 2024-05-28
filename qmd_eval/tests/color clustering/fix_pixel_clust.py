from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, clusterExtractSegs, clusterColorsToPatterns
from bigcrittercolor.projprep import clearFolder
import time
from bigcrittercolor.helpers import _getIDsInFolder

folder = "D:/bcc/ringtails"
clearFolder(folder + "/patterns",ask=False)
ids = _getIDsInFolder("D:/bcc/ringtails/segments")
ids = ids[1:2]
print(ids)
# patch
clusterColorsToPatterns(img_ids=ids, cluster_individually=True, preclustered = False, #group_cluster_records_colname = "species",
                    by_patches=False, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':4, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'find_n_minmax':(4,10), 'find_n_metric':'all', 'algo':"gaussian_mixture"}, use_positions=False, #RM FIND N METRIC
                    colorspace = "cielab",
                    height_resize = 200,
                    equalize_args={'type':"clahe"},
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "phylo_preclustered",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=False, show_indv=False, data_folder=folder)