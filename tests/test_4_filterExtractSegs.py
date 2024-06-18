import pytest
import os
import shutil
import random

from bigcrittercolor import filterExtractSegs
from bigcrittercolor.helpers import _getBCCIDs

def test_default(shared_temp_folder):
    ## CREATE DATA FOLDER TO ACT ON
    data_folder = shared_temp_folder + "/tmp"
    shutil.copytree(shared_temp_folder + "/masked", data_folder)

    filterExtractSegs(data_folder=data_folder,preselected_clusters_input=1)

    n_imgs = len(_getBCCIDs(type="segment", data_folder=data_folder))

    ## DELETE DATA FOLDER
    shutil.rmtree(data_folder)

    assert(n_imgs > 0)

def test_aux(shared_temp_folder):
    ## CREATE DATA FOLDER TO ACT ON
    data_folder = shared_temp_folder + "/tmp"
    shutil.copytree(shared_temp_folder + "/masked", data_folder)

    filterExtractSegs(used_aux_segmodel=True,preselected_clusters_input="1",data_folder=data_folder)

    n_imgs = len(_getBCCIDs(type="segment", data_folder=data_folder))

    ## DELETE DATA FOLDER
    shutil.rmtree(data_folder)

    assert (n_imgs > 0)