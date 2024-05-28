from bigcrittercolor import createBCCDataFolder, downloadiNatImageData, inferMasks, clusterExtractSegs, clusterColorsToPatterns
from bigcrittercolor.projprep import clearFolder
import time
from bigcrittercolor.helpers import _getIDsInFolder
from bigcrittercolor.projprep import showBCCImages


# this is the one I think
# patch
if __name__ == '__main__':
    folder = "D:/bcc/ringtails"

    #clearFolder(folder + "/patterns", ask=False)

    clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = False, group_cluster_records_colname = "species", # n=5
                        by_patches=True, patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':5, 'algo':'gaussian_mixture','show':False}}, visualize_patching=True,
                        cluster_args={'find_n_minmax':(4,12), 'find_n_metric':'ch', 'algo':"gaussian_mixture",'scale':'standard'}, use_positions=False, #RM FIND N METRIC
                        colorspace = "cielab",
                        height_resize = 200,
                        group_histogram_matching_colname="species",
                        #equalize_args={'type':"clahe"},
                        blur_args= {'type':"bilateral"},
                        preclust_read_subfolder = "", write_subfolder= "phylo_preclustered",
                        batch_size = None,
                        print_steps=True, print_details=False,
                        n_processes=4,
                        show=True, show_indv=False, data_folder=folder)

    clusterColorsToPatterns(img_ids=None, cluster_individually=False, preclustered = True,
                        by_patches=False,
                        cluster_args={'algo':"gaussian_mixture",'unique_values_only':True,'scale':'standard','find_n_minmax':(4,12)}, use_positions=False,
                        colorspace = "cielab",
                        blur_args= None,
                        equalize_args= None,
                        height_resize = 200,
                        preclust_read_subfolder = "phylo_preclustered",
                        batch_size = None,
                        n_processes=4,
                        print_steps=True, print_details=False,
                        show=True, show_indv=False, data_folder=folder)

    showBCCImages(data_folder=folder,sample_n=20,show_type="seg_preclust_pattern",preclust_folder_name="phylo_preclustered",
                   collage_resize_wh=(30,100))