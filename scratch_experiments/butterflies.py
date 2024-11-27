from bigcrittercolor.imgdownload import downloadiNatRandImgs
from bigcrittercolor import createBCCDataFolder, inferMasks, filterExtractSegs

#createBCCDataFolder("D:/bcc","butterflies")

#downloadiNatRandImgs(n=100,seed=40, n_before_hr_wait=100, inat_csv_location="D:/bcc/butterflies/usa_butterflies_inat_gbif.csv",sep='\t',data_folder="D:/bcc/butterflies")

inferMasks(data_folder="D:/bcc/beetles2",strategy="prompt1",text_prompt="insect",
           aux_segmodel_location="D:/bcc/beetle_appendage_segmenter/segmenter.pt",
           sam_location="D:/bcc/sam.pth", show=True,show_indv=True)

#inferMasks(data_folder="D:/bcc/beetles2",strategy="prompt1",text_prompt="leg")

#filterExtractSegs(data_folder="D:/bcc/butterflies")