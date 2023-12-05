from bigcrittercolor import createBCCDataFolder, inferMasks,  clusterExtractSegs
from bigcrittercolor.imgdownload import downloadiNatRandImgs
from bigcrittercolor import clusterColorsToPatterns

#downloadiNatRandImgs(n=200, data_folder="D:/bcc/beetles",sep="\t", inat_csv_location="D:/bcc/beetles/inat_usa_beetles.csv")
#inferMasks(skip_existing=False, data_folder="D:/bcc/beetles",show=True,show_indv=True)
#clusterExtractSegs(data_folder="D:/bcc/beetles",filter_prop_img_minmax=(0.01,0.7), color_format_to_cluster="color", show=True)

clusterColorsToPatterns(data_folder="D:/bcc/beetles",colorspace="cielab", cluster_args={'n':11, 'algo':"agglom", 'scale':True,'merge_with_user_input':True},
                        blur_args={'type':"bilateral",'d':9, 'sigma_color':150, 'sigma_space':150},show_indv=False)