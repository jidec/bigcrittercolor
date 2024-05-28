from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, clusterExtractSegs, clusterColorsToPatterns
from bigcrittercolor.projprep import clearFolder
import time

# NOTE the red bug in patterns is due to a color other than black being the dominant color

clearFolder("D:/anac_tests/patterns",ask=False)

clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False,
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'find_n_minmax':(3,7), 'algo':"gaussian_mixture"}, use_positions=False,
                    colorspace = "cielab",
                    height_resize = 200,
                    equalize_args={'type':"clahe"},
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "phylo_preclustered",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=True, data_folder="D:/anac_tests")

clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False,
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'find_n_minmax':(3,7), 'algo':"gaussian_mixture"}, use_positions=False,
                    colorspace = "cielab",
                    height_resize = 200,
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "phylo_preclustered",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="D:/anac_tests")

# whether this helps or not is complex - it appears to brighten and flesh out color patches in some cases, but it brightens in a complex way