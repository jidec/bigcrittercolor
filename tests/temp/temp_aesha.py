from bigcrittercolor.imgdownload import downloadiNatImageData
from bigcrittercolor.segmentation import inferMasks
from bigcrittercolor.helpers import _rebuildiNatRecords
from bigcrittercolor.maskfilter import filterClusterMasksExtractSegs
from bigcrittercolor.dataprep import printBCCDataStatus
from bigcrittercolor.morphcluster import inferClusterViewsMorphs

#downloadiNatGenusImages(genus_list_start_index=8,genus_list_end_index=9,
#img_size="medium",data_folder="E:/aeshna_data")

#inferMasks(img_ids=None,data_folder="E:/aeshna_data", print_details=True,
#               text_prompt="dragonfly . wing .", archetype="remove_prompt2_from1", erode_kernel_size=3, show=False)

#_rebuildiNatRecords(data_folder="E:/aeshna_data")

filterClusterMasksExtractSegs(img_ids=None, filter_hw_ratio_minmax=(3, 100), filter_prop_img_minmax=(0.01, 0.25),
                              data_folder="E:/aeshna_data",print_details=True)


#printBCCDataStatus("E:/aeshna_data")

#inferClusterViewsMorphs(img_ids=None, records_group_col="species",
#                        data_folder="E:/aeshna_data")
