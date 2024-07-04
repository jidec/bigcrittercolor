import pytest
import os
import shutil

from bigcrittercolor import downloadiNatImageData
from bigcrittercolor.helpers import _getBCCIDs, _createTempTestFolder, _runtime
from bigcrittercolor.project import convertProjectToDb

def test_default(shared_temp_folder):
    data_folder = _createTempTestFolder(shared_temp_folder,"empty")

    downloadiNatImageData(taxa_list=["Anaciaeschna"], data_folder=data_folder)

    n_downloaded_imgs = len(_getBCCIDs(type="image",data_folder=data_folder))

    assert(n_downloaded_imgs > 100) # images were downloaded because their IDs are present in the data

def test_records_before_imgs(shared_temp_folder):
    data_folder = _createTempTestFolder(shared_temp_folder,"empty")

    downloadiNatImageData(taxa_list=["Anaciaeschna"],download_records=True,download_images=False,data_folder=data_folder)
    downloadiNatImageData(taxa_list=["Anaciaeschna"], download_records=False, download_images=True,
                          data_folder=data_folder)

    n_downloaded_imgs = len(_getBCCIDs(type="image", data_folder=data_folder))

    assert (n_downloaded_imgs > 100)  # images were downloaded because their IDs are present in the data

def test_db(shared_temp_folder):
    data_folder = _createTempTestFolder(shared_temp_folder, "empty")

    convertProjectToDb(data_folder,map_size_gb=3) # convert to db first

    # otherwise same as default test
    downloadiNatImageData(taxa_list=["Anaciaeschna"], data_folder=data_folder)

    n_downloaded_imgs = len(_getBCCIDs(type="image", data_folder=data_folder))

    assert n_downloaded_imgs > 100  # images were downloaded because their IDs are present in the data

def test_skip_existing(shared_temp_folder):
    data_folder = _createTempTestFolder(shared_temp_folder, "empty")

    time_first = _runtime(downloadiNatImageData,taxa_list=["Anaciaeschna"], data_folder=data_folder)
    time_second = _runtime(downloadiNatImageData, taxa_list=["Anaciaeschna"], data_folder=data_folder)

    prop_time_first = time_first / time_second

    assert prop_time_first > 1.5


def test_skip_existing_a_few(shared_temp_folder):
    data_folder = _createTempTestFolder(shared_temp_folder, "empty")

    time_first = _runtime(downloadiNatImageData, taxa_list=["Anaciaeschna"], data_folder=data_folder)
    time_second = _runtime(downloadiNatImageData, taxa_list=["Anaciaeschna"], data_folder=data_folder)

    prop_time_first = time_first / time_second

    assert prop_time_first > 1.5


# def test_n_records_n_imgs(shared_temp_folder):
#     ## CREATE DATA FOLDER TO ACT ON
#     data_folder = shared_temp_folder + "/tmp"
#     shutil.copytree(shared_temp_folder + "/empty", data_folder)
#
#     downloadiNatImageData(taxa_list=["Anaciaeschna"], data_folder=data_folder)
#
#     # load csv and count rows
#     # count number of images - should be similar






