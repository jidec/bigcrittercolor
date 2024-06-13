import pytest
import os
import shutil

from bigcrittercolor import downloadiNatImageData
from bigcrittercolor.helpers import _getBCCIDs

def test_default(shared_temp_folder):
    ## CREATE DATA FOLDER TO ACT ON
    data_folder = shared_temp_folder + "/tmp"
    shutil.copytree(shared_temp_folder + "/empty", data_folder)

    downloadiNatImageData(taxa_list=["Anaciaeschna"], data_folder=data_folder)

    n_downloaded_imgs = len(_getBCCIDs(type="image",data_folder=data_folder))

    ## DELETE DATA FOLDER
    shutil.rmtree(data_folder)

    assert(n_downloaded_imgs > 100) # images were downloaded because their IDs are present in the data

def test_records_before_imgs(shared_temp_folder):
    ## CREATE DATA FOLDER TO ACT ON
    data_folder = shared_temp_folder + "/tmp"
    shutil.copytree(shared_temp_folder + "/empty", data_folder)

    downloadiNatImageData(taxa_list=["Anaciaeschna"],download_records=True,download_images=False,data_folder=data_folder)
    downloadiNatImageData(taxa_list=["Anaciaeschna"], download_records=True, download_images=False,
                          data_folder=data_folder)

    n_downloaded_imgs = len(_getBCCIDs(type="image", data_folder=data_folder))

    ## DELETE DATA FOLDER
    shutil.rmtree(data_folder)

    assert (n_downloaded_imgs > 100)  # images were downloaded because their IDs are present in the data