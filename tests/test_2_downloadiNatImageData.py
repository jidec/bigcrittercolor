import pytest
import os
import shutil

from bigcrittercolor import downloadiNatImageData
from bigcrittercolor.helpers import _getBCCIDs
from bigcrittercolor.project import convertProjectToDb

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
    downloadiNatImageData(taxa_list=["Anaciaeschna"], download_records=False, download_images=True,
                          data_folder=data_folder)

    n_downloaded_imgs = len(_getBCCIDs(type="image", data_folder=data_folder))

    ## DELETE DATA FOLDER
    shutil.rmtree(data_folder)

    assert (n_downloaded_imgs > 100)  # images were downloaded because their IDs are present in the data

def test_db(shared_temp_folder):
    ## CREATE DATA FOLDER TO ACT ON
    data_folder = shared_temp_folder + "/tmp"
    shutil.copytree(shared_temp_folder + "/empty", data_folder)

    convertProjectToDb(data_folder,map_size_gb=3) # convert to db first

    # otherwise same as default test
    downloadiNatImageData(taxa_list=["Anaciaeschna"], data_folder=data_folder)

    n_downloaded_imgs = len(_getBCCIDs(type="image", data_folder=data_folder))

    ## DELETE DATA FOLDER
    shutil.rmtree(data_folder)

    assert (n_downloaded_imgs > 100)  # images were downloaded because their IDs are present in the data

# def test_n_records_n_imgs(shared_temp_folder):
#     ## CREATE DATA FOLDER TO ACT ON
#     data_folder = shared_temp_folder + "/tmp"
#     shutil.copytree(shared_temp_folder + "/empty", data_folder)
#
#     downloadiNatImageData(taxa_list=["Anaciaeschna"], data_folder=data_folder)
#
#     # load csv and count rows
#     # count number of images - should be similar






