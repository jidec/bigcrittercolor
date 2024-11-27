
from bigcrittercolor import createBCCDataFolder, downloadiNatImagesAndData,inferMasks,filterExtractSegs,clusterColorsToPatterns
from bigcrittercolor.helpers import _getBCCIDs

#createBCCDataFolder(parent_folder="D:/bcc",new_folder_name="colaspidema")
#downloadiNatImageData(taxa_list=["Colaspidema"],data_folder="D:/bcc/colaspidema")
#inferMasks(data_folder="D:/bcc/colaspidema",sam_location="D:/bcc/sam.pth")
#filterExtractSegs(data_folder="D:/bcc/colaspidema")
#if __name__ == '__main__':
#    ids = _getBCCIDs(type="segment",records_filter_dict={'species':'Colaspidema dufouri'}, data_folder="D:/bcc/colaspidema")
#    clusterColorsToPatterns(height_resize=30,img_ids=ids,by_patches=False,colorspace="rgb",data_folder="D:/bcc/colaspidema",n_processes=3,group_histogram_matching_colname="species")

if __name__ == '__main__':
    clusterColorsToPatterns(height_resize=60,by_patches=False,colorspace="cielab",data_folder="D:/bcc/ringtails",n_processes=3,group_histogram_matching_colname="species")