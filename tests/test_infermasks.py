from bigcrittercolor.segment import inferMasks
import os

def test_infermasks():
    inferMasks(img_ids=None,skip_existing=False,data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder",
               text_prompt="dragonfly", strategy="prompt1", erode_kernel_size=3, show=True, show_indv=True)
#test_infermasks()

#text_prompt="dragonfly . wing ."
#strategy="remove_prompt2_from1"

def test_infermasksindv():
    ids = ["INAT-1828591-1"]
    inferMasks(img_ids=ids, skip_existing=False,
               text_prompt="dragonfly . wing .", strategy="remove_prompt2_from1",
               erode_kernel_size=3, remove_islands=True,
               show_indv=True, print_steps=True, print_details=True,
               data_folder="E:/aeshna_data")

def test_inferMasksCpu():
    inferMasks(gpu=False, img_ids=None,skip_existing=False,data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder",
               text_prompt="dragonfly", strategy="prompt1", erode_kernel_size=3, show=True, show_indv=True,print_details=True)

test_inferMasksCpu()
