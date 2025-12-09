import pytest
import os
import shutil
import random

from bigcrittercolor import filterExtractSegs
from bigcrittercolor.helpers import _getBCCIDs, _createTempTestFolder

def test_default(shared_temp_folder):
    data_folder = _createTempTestFolder(shared_temp_folder, "masked")

    filterExtractSegs(data_folder=data_folder,preselected_clusters_input="1")

    n_imgs = len(_getBCCIDs(type="segment", data_folder=data_folder))

    assert(n_imgs > 0)

def test_aux(shared_temp_folder):
    data_folder = _createTempTestFolder(shared_temp_folder, "masked")

    filterExtractSegs(used_aux_segmodel=True,preselected_clusters_input="1",data_folder=data_folder)

    n_imgs = len(_getBCCIDs(type="segment", data_folder=data_folder))

    assert (n_imgs > 0)

def test_batch(shared_temp_folder):
    data_folder = _createTempTestFolder(shared_temp_folder, "masked")

    filterExtractSegs(used_aux_segmodel=True,batch_size=50,preselected_clusters_input="1",data_folder=data_folder)

    n_imgs = len(_getBCCIDs(type="segment", data_folder=data_folder))

    assert (n_imgs > 0)