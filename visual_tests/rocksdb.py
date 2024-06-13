import cv2
#import rocksdbpy

from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, filterExtractSegs, clusterColorsToPatterns, writeMetricsFromPatterns
from bigcrittercolor.helpers import _getBCCIDs, _readBCCImgs
from bigcrittercolor.helpers.db import _readDb
from bigcrittercolor.project import convertProjectToDb

folder_name = "anacs_db"
data_folder = "D:/bcc/" + folder_name

#convertProjectToDb(data_folder=data_folder)

#createBCCDataFolder(parent_folder="D:/bcc", new_folder_name=folder_name,use_db=True)

#downloadiNatImageData(taxa_list=["Anaciaeschna"],data_folder=data_folder)

#inferMasks(data_folder=data_folder,gd_gpu=True,sam_gpu=True,sam_location="D:/bcc/sam.pth",
#              aux_segmodel_location="D:/bcc/aux_segmenter.pt",print_details=True)

# filterExtractSegs(used_aux_segmodel=True,
#                       filter_prop_img_minmax=None, cluster_params_dict={'pca_n':8}, filter_intersects_sides= None, filter_symmetry_min = None,
#                       filter_hw_ratio_minmax=None,
#                       feature_extractor="resnet18",data_folder=data_folder)

if __name__ == '__main__':
    clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered=False,
                            group_cluster_records_colname=None,#"species",  # n=5
                            by_patches=True, patch_args={'min_patch_pixel_area': 5,
                                                         'cluster_args': {'n': 5, 'algo': 'gaussian_mixture',
                                                                          'show': False}}, visualize_patching=False,
                            cluster_args={'find_n_minmax': (4, 12), 'find_n_metric': 'ch', 'algo': "gaussian_mixture",
                                          'scale': 'standard'}, use_positions=False,  # RM FIND N METRIC
                            colorspace="cielab",
                            height_resize=200,
                            group_histogram_matching_colname="species",
                            # equalize_args={'type':"clahe"},
                            blur_args={'type': "bilateral"},
                            preclust_read_subfolder="", write_subfolder="phylo_preclustered",
                            batch_size=None,
                            print_steps=True, print_details=False,
                            n_processes=4,
                            show=True, show_indv=False, data_folder=data_folder)
if __name__ == '__main__':
    clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered=True,
                            by_patches=False,
                            cluster_args={'algo': "gaussian_mixture", 'unique_values_only': True, 'scale': 'standard',
                                          'find_n_minmax': (4, 12)}, use_positions=False,
                            colorspace="cielab",
                            blur_args=None,
                            equalize_args=None,
                            height_resize=200,
                            preclust_read_subfolder="phylo_preclustered",
                            batch_size=None,
                            n_processes=4,
                            print_steps=True, print_details=False,
                            show=True, show_indv=False, data_folder=data_folder)
