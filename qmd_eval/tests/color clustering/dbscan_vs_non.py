from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, clusterExtractSegs, clusterColorsToPatterns
from bigcrittercolor.projprep import clearFolder
import time


clearFolder("D:/anac_tests/patterns",ask=False)
# Record the start time
start_time = time.time()

clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False, group_cluster_records_colname = "species",
                    group_histogram_matching_colname="species",
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'find_n_minmax':(3,10), 'algo':"gaussian_mixture"}, use_positions=False,
                    colorspace = "cielab",
                    height_resize = 200,
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "", write_subfolder= "phylo_preclustered",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="D:/anac_tests")

# Record the end time
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time
print("Total execution time:", total_time, "seconds")

clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = True,
                    by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture'}}, visualize_patching=True,
                    cluster_args={'algo':"hdbscan"}, use_positions=False,
                    colorspace = "cielab",
                    height_resize = 200,
                    blur_args= {'type':"bilateral"},
                    preclust_read_subfolder = "phylo_preclustered",
                    batch_size = None,
                    print_steps=True, print_details=False,
                    show=True, show_indv=False, data_folder="D:/anac_tests")