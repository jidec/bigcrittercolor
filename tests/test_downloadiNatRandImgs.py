from bigcrittercolor.imgdownload.downloadiNatRandImgs import downloadiNatRandImgs

def test_downloadrand():
    downloadiNatRandImgs(n=3, seed=30, data_folder="D:/GitProjects/bigcrittercolor/tests/dummy_bcc_folder", sep="\t",
                         inat_csv_location="E:/dragonfly-patterner/data/other/raw_records/inatdragonflyusa.csv")
test_downloadrand()