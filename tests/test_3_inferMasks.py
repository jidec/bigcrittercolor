import pytest
import os
import shutil
import random

from bigcrittercolor import inferMasks
from bigcrittercolor.helpers import _getBCCIDs, _createTempTestFolder

def test_default(shared_temp_folder):
    data_folder = _createTempTestFolder(shared_temp_folder, "downloaded")

    # get 3 IDs, inferring for hundreds will take too long
    ids = _getBCCIDs(type="image",data_folder=data_folder)
    ids = random.sample(ids, 3)

    inferMasks(img_ids=ids,data_folder=data_folder,sam_location="D:/bcc/sam.pth",
                aux_segmodel_location="D:/bcc/aux_segmenter.pt")

    n_masks = len(_getBCCIDs(type="mask", data_folder=data_folder))

    assert (n_masks == 3)  # images are present because their IDs are present in the data

