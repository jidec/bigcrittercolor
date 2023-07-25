from bigcrittercolor.dataprep import writeiNatGenusList
import os

def test_writeinat():
    writeiNatGenusList(inat_csv_name="dummy_inat_records", data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_data_folder")
    #assert os.path.exists("D:/GitProjects/bigcrittercolor/tests/dummy_data_folder/other/raw_records/inat_genus_list.csv"), "Test failed"
test_writeinat()