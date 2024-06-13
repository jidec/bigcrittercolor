from bigcrittercolor import filterExtractSegs

#filterExtractSegs(data_folder="D:/bcc/beetles",cluster_algo="affprop")
#filterExtractSegs(data_folder="D:/bcc/wake_raleigh", cluster_algo="affprop", cluster_params_dict={'preference':-500})
#filterExtractSegs(data_folder=)


filterExtractSegs(feature_extractor="inceptionv3", cluster_algo="affprop", cluster_params_dict={'preference':-500},
                        show=True, show_indv=False,
                        print_steps=True, data_folder='D:/bcc/wake_raleigh')