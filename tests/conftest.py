import pytest
import os
import shutil
from tempfile import TemporaryDirectory
import random

from bigcrittercolor import createBccDataFolder, downloadiNatImagesAndData, inferMasks, filterExtractSegments
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

#pytest --permanent-folder="D:/GitProjects/bigcrittercolor/tests/intermediate_data_folders" -s
#pytest D:/GitProjects/bigcrittercolor/tests/test_4_filterExtractSegs.py::test_batch --permanent-folder="D:/GitProjects/bigcrittercolor/tests/intermediate_data_folders" -s
@pytest.fixture(scope="session")
def shared_temp_folder(request):
    # whether to use a permanent folder, which allows you save time running tests
    permanent_folder = request.config.getoption("--permanent-folder")

    if permanent_folder:
        temp_folder = permanent_folder
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
    else:
        temp_folder = TemporaryDirectory().name
        os.makedirs(temp_folder)

    print("Using temp directory for test session:", temp_folder)

    # All temporary folders
    empty_folder = os.path.join(temp_folder, "empty")
    downloaded_folder = os.path.join(temp_folder, "downloaded")
    masked_folder = os.path.join(temp_folder, "masked")
    filtered_folder = os.path.join(temp_folder, "filtered")
    masked_auxseg_folder = os.path.join(temp_folder, "masked_auxseg")
    filtered_auxseg_folder = os.path.join(temp_folder, "filtered_auxseg")

    if not os.path.exists(empty_folder):
        # Create folders at intermediate stages of the pipeline
        # Empty
        createBccDataFolder(parent_folder=temp_folder, new_folder_name="empty")

        # Downloaded (Anaciaeschna)
        shutil.copytree(empty_folder, downloaded_folder)
        downloadiNatImagesAndData(taxa_list=["Anaciaeschna"], data_folder=downloaded_folder)

        # Normal - not using auxseg
        # Masked
        shutil.copytree(downloaded_folder, masked_folder)
        # Get 30 IDs, inferring for hundreds will take too long
        ids = _getBCCIDs(type="image", data_folder=masked_folder)
        ids = random.sample(ids, 30)
        inferMasks(img_ids=ids, sam_location="D:/bcc/sam.pth", data_folder=masked_folder)

        # Filtered
        shutil.copytree(masked_folder, filtered_folder)
        filterExtractSegs(used_aux_segmodel=False, preselected_clusters_input="1", data_folder=filtered_folder)

        # Aux - using auxseg
        # Masked auxseg
        shutil.copytree(downloaded_folder, masked_auxseg_folder)
        # Get 30 IDs, inferring for hundreds will take too long
        ids = _getBCCIDs(type="image", data_folder=masked_auxseg_folder)
        ids = random.sample(ids, 30)
        inferMasks(img_ids=ids, data_folder=masked_auxseg_folder, sam_location="D:/bcc/sam.pth",
                   aux_segmodel_location="D:/bcc/aux_segmenter.pt")

        # Filtered auxseg
        shutil.copytree(masked_auxseg_folder, filtered_auxseg_folder)
        filterExtractSegs(used_aux_segmodel=True, preselected_clusters_input="1", data_folder=filtered_auxseg_folder)

    yield temp_folder

    if not permanent_folder:
        shutil.rmtree(temp_folder)  # Cleanup when using temporary folder

def pytest_addoption(parser):
    parser.addoption("--permanent-folder", action="store", default=None, help="Path to a permanent folder for testing")