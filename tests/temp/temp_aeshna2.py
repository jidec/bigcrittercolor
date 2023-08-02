from bigcrittercolor.segmentation import inferMasks
from bigcrittercolor.helpers import _getIDsInFolder
from bigcrittercolor.helpers import showBCCImages
import random

# get 100 random ids for experiment
ids = _getIDsInFolder("D:/dfly_seg_expr/animal_prompt/all_images")
random.seed(30)
ids = random.sample(ids,100)

#inferMasks(img_ids=ids,strategy="prompt1", text_prompt="insect", data_folder="D:/dfly_seg_expr/insect_prompt",print_details=True)
inferMasks(img_ids=ids,strategy="prompt1", text_prompt="dragonfly", data_folder="D:/dfly_seg_expr/dragonfly_prompt",print_details=True)