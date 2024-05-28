from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, clusterExtractSegs, clusterColorsToPatterns
from bigcrittercolor.projprep import clearFolder
import time


clearFolder("D:/anac_tests/segments")
# Record the start time
start_time = time.time()

clusterExtractSegs(color_format_to_cluster = "grey", used_aux_segmodel=True,
    filter_hw_ratio_minmax = (3,10000), filter_prop_img_minmax = None, filter_symmetry_min = None, filter_intersects_sides=False, # filters
    mask_normalize_params_dict={'lines_strategy':"ellipse"},
    feature_extractor="resnet18", # feature extractor
    cluster_algo="kmeans", cluster_n = 4,
    cluster_params_dict={'eps':0.1,'min_samples':24}, preselected_clusters_input = "1",
    show=True, show_indv=False, print_steps=True, data_folder="D:/anac_tests")

# Record the end time
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time
print("Total execution time:", total_time, "seconds")

# Record the start time
start_time = time.time()

clusterExtractSegs(color_format_to_cluster = "grey", used_aux_segmodel=True,
    filter_hw_ratio_minmax = (3,10000), filter_prop_img_minmax = None, filter_symmetry_min = None, filter_intersects_sides=False, # filters
    mask_normalize_params_dict={'lines_strategy':"ellipse"},
    feature_extractor="resnet18", # feature extractor
    cluster_algo="kmeans", cluster_n = 4,
    cluster_params_dict={'eps':0.1,'min_samples':24,'pca_n':5}, preselected_clusters_input = "1",
    show=True, show_indv=False, print_steps=True, data_folder="D:/anac_tests")

# Record the end time
end_time = time.time()
# Calculate the total time taken
total_time = end_time - start_time
print("Total execution time:", total_time, "seconds")

# PCA at n=5 is a huge improvement to clustering quality, and even grabs views well