from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, clusterExtractSegs, clusterColorsToPatterns
from bigcrittercolor.projprep import clearFolder
import time


clearFolder("D:/anac_tests/patterns")
# Record the start time
start_time = time.time()

clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False, group_cluster_records_colname = None,
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'find_n_minmax':3, 'algo':'kmeans'}}, visualize_patching=True,
                    colorspace = "cielab",
                    height_resize = 200,
                    equalize_args={'type':"clahe"},
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="D:/anac_tests")

# Record the end time
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time
print("Total execution time:", total_time, "seconds")

# Record the start time
start_time = time.time()

clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False, group_cluster_records_colname = None,
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'find_n_minmax':(3,5), 'algo':'kmeans'}}, visualize_patching=True,
                    colorspace = "rgb",
                    height_resize = 200,
                    equalize_args={'type':"clahe"},
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="D:/anac_tests")

# Record the end time
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time
print("Total execution time:", total_time, "seconds")