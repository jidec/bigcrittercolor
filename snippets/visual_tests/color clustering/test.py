from bigcrittercolor import downloadiNatImageData
from bigcrittercolor import createBCCDataFolder
from bigcrittercolor import inferMasks

#createBCCDataFolder("D:/bcc","ringtails")
#downloadiNatImageData(taxa_list=["Erpetogomphus"],data_folder="D:/bcc/ringtails")
inferMasks(aux_segmodel_location="D:/anac_tests/other/ml_checkpoints/aux_segmenter.pt",
           sam_location="D:/anac_tests/other/ml_checkpoints/sam.pth", data_folder= "D:/bcc/ringtails")