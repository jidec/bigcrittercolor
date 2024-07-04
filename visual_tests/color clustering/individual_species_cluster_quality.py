from bigcrittercolor import clusterColorsToPatterns
from bigcrittercolor.project import clearFolder
from bigcrittercolor.helpers import _getBCCIDs

# lampropeltis
# designatus
# crotalinus
if __name__ == '__main__':
    ids = _getBCCIDs(type="segment",data_folder="D:/bcc/ringtails",records_filter_dict={'species':"Erpetogomphus lampropeltis"})
    clusterColorsToPatterns(img_ids=ids, colorspace="cielab",
                            by_patches=True, patch_args = {'min_patch_pixel_area':5}, visualize_patching=True,
                            cluster_args={'algo':"hdbscan",'show_silhouette':False,
                                'show_color_centroids':True,'show_color_scatter':True},
                            write_subfolder= "1species", n_processes=4, show_indv=False, show=False, data_folder="D:/bcc/ringtails")