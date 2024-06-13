import pytest
import os

from bigcrittercolor import createBCCDataFolder
def test_default(shared_temp_folder):
    createBCCDataFolder(parent_folder=shared_temp_folder) # doesn't error out
    assert os.path.exists(shared_temp_folder + "/bcc_data") # folder was created