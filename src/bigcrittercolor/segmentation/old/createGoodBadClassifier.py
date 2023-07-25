import random
from bigcrittercolor.segmentation import inferMasks
from bigcrittercolor.helpers import _getIDsInFolder
from bigcrittercolor.maskfilter import filterClusterMasksExtractSegs

# this function would:
# 1. use inferMasks to infer nmasks masks
# 2. filterCluster the masks for those with good segs
# 3. move good segs and bad segs to training directories
# 4. train and test a classifier
# 5. place the classifier so it can be used as a filter in subsequent inferMasks

def createInitialFilter(seg_text_prompt, training_set_size=1000, data_folder=''):
    # get all downloaded image ids
    all_img_ids = _getIDsInFolder(data_folder + "/all_images")
    # sample random ids
    sampled_ids= random.sample(all_img_ids, training_set_size)

    # infer masks for those ids, they get put in masks folder
    inferMasks(img_ids=sampled_ids,data_folder=data_folder,
               text_prompt="dragonfly . wing .", archetype="remove_prompt2_from1", erode_kernel_size=3, show=False)

    filterClusterMasksExtractSegs(img_ids=sampled_ids, filter_hw_ratio_minmax=(3, 100), filter_prop_img_minmax=(0, 0.25),
                                  data_folder=data_folder)

    print(0)