from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, filterExtractSegs, clusterColorsToPatterns
from bigcrittercolor.project import clearFolder
import time

# NOTE the red bug in patterns is due to a color other than black being the dominant color

clearFolder("D:/anac_tests/patterns",ask=False)

clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False, group_cluster_records_colname = "species",
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'find_n_minmax':(3,10), 'algo':"gaussian_mixture"}, use_positions=False,
                    colorspace = "cielab",
                    height_resize = 200,
                    equalize_args={'type':"clahe"},
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "phylo_preclustered",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="D:/anac_tests")

#clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False, group_cluster_records_colname = None,
#                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
#                    cluster_args={'find_n_minmax':(3,10), 'algo':"gaussian_mixture"}, use_positions=False,
#                    colorspace = "cielab",
#                    height_resize = 200,
#                    equalize_args={'type':"clahe"},
#                    blur_args= {'type':"bilateral"},
#                    preclust_read_subfolder = "", write_subfolder= "",
#                    batch_size = None,
#                    print_steps=True, print_details=False,
#                    show=True, show_indv=False, data_folder="D:/anac_tests")

# the phylo reclustering
clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = True,
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'n':6, 'algo':"gaussian_mixture",'unique_values_only':True}, use_positions=False, #'find_n_minmax':(3,10)
                    colorspace = "cielab",
                    height_resize = 200,
                    equalize_args= None,
                    blur_args= None,
                    preclust_read_subfolder = "phylo_preclustered",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="D:/anac_tests")