from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, filterExtractSegs, clusterColorsToPatterns
from bigcrittercolor.project import clearFolder
import time

#scale is causing problems for some reason, maybe its changing the actual values
clearFolder("D:/anac_tests/patterns",ask=False)

clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False, group_cluster_records_colname = None,
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'find_n_minmax':(3,10), 'algo':"gaussian_mixture",'scale':"minmax"}, use_positions=False,
                    colorspace = "cielab",
                    height_resize = 200,
                    equalize_args={'type':"clahe"},
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="D:/anac_tests")

clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False, group_cluster_records_colname = None,
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'find_n_minmax':(3,10), 'algo':"gaussian_mixture",'scale':False}, use_positions=False,
                    colorspace = "cielab",
                    height_resize = 200,
                    equalize_args={'type':"clahe"},
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="D:/anac_tests")