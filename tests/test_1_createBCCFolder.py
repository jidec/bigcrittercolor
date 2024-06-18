import pytest
import os

from bigcrittercolor import createBCCDataFolder

#_default tests are the most basic use case using default parameters
def test_default(shared_temp_folder):
    createBCCDataFolder(parent_folder=shared_temp_folder) # doesn't error out
    assert os.path.exists(shared_temp_folder + "/bcc_data") # folder was created

def test_db(shared_temp_folder):
    createBCCDataFolder(parent_folder=shared_temp_folder,use_db=True,map_size_gb=5)  # doesn't error out
    assert os.path.exists(shared_temp_folder + "/bcc_data/db")  # folder was created