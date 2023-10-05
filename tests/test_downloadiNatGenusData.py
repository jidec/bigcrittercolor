from bigcrittercolor.imgdownload import downloadiNatImageData
from bigcrittercolor.projprep import createBCCDataFolder
import os

# note that if you download records as a size then repeat the download with a different size the old size will be kept
#   unless you delete the records folders
def test_downloadinatgenusimgs():
    genus_list = ["Stenogomphurus"]
    downloadiNatImageData(taxa_list = genus_list, img_size="medium",data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder")
#test_downloadinatgenusimgs()

def test_downloadiNatImageData_species():
    createBCCDataFolder("D:/", "anole_data")
    species_list = ["Anolis carolinensis"]
    downloadiNatImageData(taxa_list = species_list, img_size="medium",data_folder="D:/anole_data")

#test_downloadiNatImageData_species()
