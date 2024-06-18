import pytest
import os
import shutil
from tempfile import TemporaryDirectory
import random

from bigcrittercolor import createBCCDataFolder,downloadiNatImageData,inferMasks,filterExtractSegs
from bigcrittercolor.helpers import _getBCCIDs

# tests in bigcrittercolor are at the level of the core steps in the base module
# they assert that the steps run without error and produce valid outputs given different combinations of parameters
# in conftest we create a temporary directory that holds bigcrittercolor data folders that the steps apply to, and yield it to all the tests
# also in conftest and within the tmpdir, we create folders representing intermediate steps of the pipeline
# then, each test copies an intermediate folder, applies a function to the copy, then deletes the copy
# this way we don't have to constantly rebuild new folders
# these tests all apply to a small dragonfly genus, "Anaciaeschna"
# note that they are NOT visual, just checking that functions don't error out and that the basic properties of their outputs are correct
# the "visual_tests" folder contains visual tests for the clustering stages

#pytest -s

@pytest.fixture(scope="session")
def shared_temp_folder():
    with TemporaryDirectory() as temp_folder:
        print("Creating temp directory for test session:", temp_folder)

        # all temporary folders
        empty_folder = temp_folder + "/empty"
        downloaded_folder = temp_folder + "/downloaded"
        masked_folder = temp_folder + "/masked"
        filtered_folder = temp_folder + "/filtered"
        masked_auxseg_folder = temp_folder + "/masked_auxseg"
        filtered_auxseg_folder = temp_folder + "/filtered_auxseg"

        # create folders at intermediate stages of the pipeline
        # empty
        createBCCDataFolder(parent_folder=temp_folder,new_folder_name="empty")

        # downloaded (Anaciaeschna)
        shutil.copytree(empty_folder, downloaded_folder)
        downloadiNatImageData(taxa_list=["Anaciaeschna"], data_folder=downloaded_folder)

        ## normal - not using auxseg
        # masked
        shutil.copytree(downloaded_folder, masked_folder)
        # get 30 IDs, inferring for hundreds will take too long
        ids = _getBCCIDs(type="image", data_folder=masked_folder)
        ids = random.sample(ids, 30)
        inferMasks(img_ids=ids, sam_location="D:/bcc/sam.pth",
                   data_folder=masked_folder)
        # filtered
        shutil.copytree(masked_folder, filtered_folder)
        filterExtractSegs(used_aux_segmodel=False, preselected_clusters_input="1", data_folder=filtered_folder)

        ## aux - using auxseg
        # masked auxseg
        shutil.copytree(downloaded_folder, masked_auxseg_folder)
        # get 30 IDs, inferring for hundreds will take too long
        ids = _getBCCIDs(type="image", data_folder=masked_auxseg_folder)
        ids = random.sample(ids, 30)
        inferMasks(img_ids=ids, data_folder=masked_auxseg_folder, sam_location="D:/bcc/sam.pth",
                   aux_segmodel_location="D:/bcc/aux_segmenter.pt")
        # filtered auxseg
        shutil.copytree(masked_auxseg_folder, filtered_auxseg_folder)
        filterExtractSegs(used_aux_segmodel=True,preselected_clusters_input="1",data_folder=filtered_auxseg_folder)
        
        yield temp_folder
        # cleanup happens automatically when the block exits