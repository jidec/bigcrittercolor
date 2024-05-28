from bigcrittercolor import clusterColorsToPatterns
from bigcrittercolor.helpers import _getIDsInFolder
import random

ids = _getIDsInFolder("E:/aeshna_data/patterns/taxa_preclust")
clusterColorsToPatterns(img_ids=ids,data_folder="E:/aeshna_data",colorspace="rgb", #group_cluster_records_colname="species",
                        patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':4, 'algo':'gaussian_mixture'}}, #find_n_minmax':(3,7)
                        cluster_args={'n':15, 'algo':"kmeans", 'scale':True,'linkage':"complete",'find_n_minmax':(3,7)}, #'merge_with_user_input':True,
                        blur_args={'type':"bilateral",'d':9, 'sigma_color':150, 'sigma_space':150,'auto_adjust_blur':True},
                        equalize_args=None, show_indv=False,preclust_read_subfolder="taxa_preclust")

# patched images with cielab are all one color
# however you can do cielab and _format it in _ToPatches and it kinda works - close I think the problem is when patch data is cielab formatted or something
ids = _getIDsInFolder("E:/aeshna_data/segments")
ids = random.sample(ids,5000)
clusterColorsToPatterns(img_ids=ids,data_folder="E:/aeshna_data",colorspace="cielab", #cielab
                        patch_args = {'min_patch_pixel_area':5,'cluster_args':{'n':4, 'algo':'gaussian_mixture'}}, #find_n_minmax':(3,7)
                        cluster_args={'n':15, 'algo':"kmeans", 'scale':True,'linkage':"complete",'find_n_minmax':(3,7)}, #'merge_with_user_input':True,
                        blur_args={'type':"bilateral",'d':9, 'sigma_color':150, 'sigma_space':150,'auto_adjust_blur':True},
                        equalize_args=None, show_indv=False)