from bigcrittercolor.segmentation import inferMasksFast
from bigcrittercolor.segmentation import inferMasks
import os

def test_infermasksfast():
    inferMasks(img_ids=None,data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder",
               text_prompt="dragonfly", archetype="prompt1", erode_kernel_size=3, print_steps=True, print_details=False, show_indv=False)
test_infermasksfast()