from bigcrittercolor import clusterExtractSegs

#clusterExtractSegs(data_folder="D:/bcc/beetles",cluster_algo="affprop")
#clusterExtractSegs(data_folder="D:/bcc/wake_raleigh", cluster_algo="affprop", cluster_params_dict={'preference':-500})
#clusterExtractSegs(data_folder=)


clusterExtractSegs(feature_extractor="inceptionv3", cluster_algo="affprop", cluster_params_dict={'preference':-500},
                        show=True, show_indv=False,
                        print_steps=True, data_folder='D:/bcc/wake_raleigh')