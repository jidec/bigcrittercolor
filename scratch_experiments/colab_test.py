from bigcrittercolor import createBccDataFolder

#createBccDataFolder(new_folder_name="my_taxon",parent_folder="D:/bcc")

from bigcrittercolor import downloadiNatImagesAndData

#downloadiNatImagesAndData(taxa_list=["Galerita"], download_n_per_species=200, data_folder="D:/bcc/my_taxon")

#from bigcrittercolor import inferMasks

#inferMasks(sam_location="D:/bcc/sam.pth",data_folder="D:/bcc/my_taxon")

#from bigcrittercolor import filterExtractSegments

#filterExtractSegments(data_folder="D:/bcc/my_taxon")

#from bigcrittercolor import trainBioEncoder

#from bigcrittercolor.project import setupBioencoderTrainingFolder

#setupBioencoderTrainingFolder(min_imgs_per_class=20,data_folder="D:/bcc/my_taxon")

from bigcrittercolor import trainBioEncoder

trainBioEncoder(data_folder="D:/bcc/my_taxon")



