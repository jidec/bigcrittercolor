from bigcrittercolor import clusterColorsToPatterns
from bigcrittercolor.project import clearFolder
from bigcrittercolor.helpers import _getBCCIDs


if __name__ == '__main__':
    #ids = _getBCCIDs(type="segment",data_folder="D:/bcc/ringtails",records_filter_dict={'species':"Erpetogomphus designatus"})
    clusterColorsToPatterns(img_ids=None,
                            by_patches=True, patch_args = {'min_patch_pixel_area':5},
                            cluster_args={'find_n_minmax':(4,10), 'find_n_metric':'ch', 'algo':"gaussian_mixture",'scale':'standard','show_color_scatter':True},
                            write_subfolder= "1species", n_processes=4, data_folder="D:/bcc/ringtails")