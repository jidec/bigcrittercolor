from bigcrittercolor import createBCCDataFolder, inferMasks,  clusterExtractSegs

clusterExtractSegs(data_folder="D:/bcc/beetles",filter_prop_img_minmax=(0.01,0.7), preselected_clusters_input="0,1", show=True)