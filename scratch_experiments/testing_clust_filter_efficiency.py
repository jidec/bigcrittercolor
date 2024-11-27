from bigcrittercolor import filterExtractSegs

# filterExtractSegs(sample_n=None, batch_size=None,
#     color_format_to_cluster = "grey", used_aux_segmodel=True,
#     filter_hw_ratio_minmax = None, filter_prop_img_minmax = None, filter_symmetry_min = None, filter_not_intersects_sides= False, # standard img processing filters
#     mask_normalize_params_dict={'lines_strategy':"ellipse"}, # normalization/verticalization of masks
#     feature_extractor="mobilenet_v2", # feature extractor
#     cluster_params_dict={'algo':"kmeans",'pca_n':5,'n':4,'scale':"standard"}, preselected_clusters_input = None,
#     hist_cluster_params_dict = {'algo':"agglom"},
#     illum_outliers_percent = None,
#     show=True, show_save=True, show_indv=False, print_steps=True, data_folder="D:/bcc/ringtails")

filterExtractSegs(data_folder="D:/bcc/ringtails",illum_outliers_percent=0.1,used_aux_segmodel=True,show=True,show_indv=True)