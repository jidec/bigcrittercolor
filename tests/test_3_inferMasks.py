import pytest
import os
import shutil
import random

from bigcrittercolor import inferMasks
from bigcrittercolor.helpers import _getBCCIDs

def test_default(shared_temp_folder):
    ## CREATE DATA FOLDER TO ACT ON
    data_folder = shared_temp_folder + "/tmp"
    shutil.copytree(shared_temp_folder + "/downloaded", data_folder)

    # get 30 IDs, inferring for hundreds will take too long
    ids = _getBCCIDs(type="image",data_folder=data_folder)
    ids = random.sample(ids, 3)

    inferMasks(img_ids=ids,data_folder=data_folder,gd_gpu=True,sam_gpu=True,sam_location="D:/bcc/sam.pth",
                aux_segmodel_location="D:/bcc/aux_segmenter.pt",print_details=True)

    ## DELETE DATA FOLDER
    shutil.rmtree(data_folder)
