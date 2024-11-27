#from bigcrittercolor import createBigcrittercolorDataFolder

#createBigcrittercolorDataFolder(parent_folder="D:/bcc",new_folder_name="widows")

#from bigcrittercolor import downloadiNatImagesAndData

#downloadiNatImagesAndData(taxa_list=["Latrodectus tredecimguttatus"],data_folder="D:/bcc/widows")

from bigcrittercolor import downloadImagesUsingDarwinCore

downloadImagesUsingDarwinCore(dwc_archive_location="C:/Users/hiest/Desktop/Absolutely Key Papers/new_wings_downloads/beetles_inat_obsorg_dwc.zip",data_folder="D:/bcc/widows")

#from bigcrittercolor import inferMasks

#inferMasks(sam_location="D:/bcc/sam.pth",data_folder="D:/bcc/widows",show_indv=True)

#from bigcrittercolor import filterExtractSegs

#filterExtractSegs(data_folder="D:/bcc/widows")

