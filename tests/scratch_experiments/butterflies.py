from bigcrittercolor.imgdownload import downloadiNatRandImgs
from bigcrittercolor import createBCCDataFolder, inferMasks, clusterExtractSegs

createBCCDataFolder("D:/bcc","butterflies")

downloadiNatRandImgs(n=100,seed=40, n_before_hr_wait=100, inat_csv_location="D:/bcc/butterflies/usa_butterflies_inat_gbif.csv",sep='\t',data_folder="D:/bcc/butterflies")

inferMasks(data_folder="D:/bcc/butterflies")

clusterExtractSegs(data_folder="D:/bcc/butterflies")