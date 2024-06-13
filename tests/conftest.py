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

@pytest.fixture(scope="session")
def shared_temp_folder():
    with TemporaryDirectory() as temp_folder:
        print("Creating temp directory for test session:", temp_folder)

        # create folders at intermediate stages of the pipeline
        # empty
        createBCCDataFolder(parent_folder=temp_folder,new_folder_name="empty")

        # downloaded
        shutil.copytree(temp_folder + "/empty", temp_folder + "/downloaded")
        downloadiNatImageData(taxa_list=["Anaciaeschna"], data_folder=temp_folder + "/downloaded")

        # masked
        shutil.copytree(temp_folder + "/downloaded", temp_folder + "/masked")
        # get 30 IDs, inferring for hundreds will take too long
        ids = _getBCCIDs(type="image", data_folder=temp_folder + "/masked")
        ids = random.sample(ids, 30)
        inferMasks(img_ids=ids, data_folder=temp_folder + "/masked", gd_gpu=True, sam_gpu=True, sam_location="D:/bcc/sam.pth",
                   aux_segmodel_location="D:/bcc/aux_segmenter.pt")

        # filtered
        shutil.copytree(temp_folder + "/masked", temp_folder + "/filtered")
        filterExtractSegs(used_aux_segmodel=True,
                             filter_prop_img_minmax=None, cluster_params_dict={'pca_n':8}, filter_intersects_sides= None, filter_symmetry_min = None,
                             filter_hw_ratio_minmax=None, preselected_clusters_input="2",
                             feature_extractor="resnet18",data_folder=temp_folder + "/filtered")
        
        yield temp_folder
        # cleanup happens automatically when the block exits