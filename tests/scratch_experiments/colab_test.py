#from bigcrittercolor import createBCCDataFolder

#createBCCDataFolder(new_folder_name="my_taxon",parent_folder="D:/bcc")

#from bigcrittercolor import downloadiNatImageData

#downloadiNatImageData(taxa_list=["Cylindrophora"], data_folder="D:/bcc/my_taxon")

#from bigcrittercolor import inferMasks
#from bigcrittercolor.helpers import _getIDsInFolder
#import os

#files = os.listdir("/content/my_taxon/all_images")
#ids = [f.replace('.jpg','') for f in files]
#ids = ids[1:50]
#inferMasks(data_folder="D:/bcc/my_taxon",print_details=True)#,img_ids=ids)

#from bigcrittercolor import clusterExtractSegs

#files = os.listdir("/content/my_taxon/masks")
#ids = [f.replace('.jpg','') for f in files]
#ids = ids[1:30]
#clusterExtractSegs(data_folder="D:/bcc/my_taxon")

from bigcrittercolor import clusterColorsToPatterns

clusterColorsToPatterns(data_folder="D:/bcc/my_taxon",colorspace="rgb")
