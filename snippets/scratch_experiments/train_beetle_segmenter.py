from bigcrittercolor.segment import trainSegModel
from bigcrittercolor import createBccDataFolder
from bigcrittercolor.imgdownload import downloadiNatRandImgs
from bigcrittercolor.helpers import _readBCCImgs, _showImages

trainSegModel(training_dir_location="D:/bcc/beetle_appendage_segmenter",num_epochs=15)

#createBCCDataFolder(parent_folder="D:/bcc",new_folder_name="all_beetles_download")

#downloadiNatRandImgs(23, n_before_hr_wait=100, sep=",", inat_csv_location="D:/bcc/inat_usa_dragonflies.csv", print_steps=True, data_folder="D:/bcc/new_random_dragonflies")

#inferMasks(skip_existing=False,sam_location="D:/bcc/sam.pth", aux_segmodel_location="D:/bcc/aux_segmenter.pt",
#           data_folder="D:/bcc/new_random_dragonflies",show_indv=True,show=True, text_prompt="dragonfly")

#aux_segmodel_location="D:/bcc/beetle_appendage_segmenter/segmenter.pt"
#filterExtractSegs(used_aux_segmodel=True,
#                  cluster_params_dict={'algo':"kmeans",'pca_n':5,'n':4,'scale':"standard"},data_folder="D:/bcc/new_random_dragonflies")

#imgs = _readBCCImgs(type="raw_segment",data_folder="D:/bcc/new_random_beetles")
#print(len(imgs))
#chunk_size = 10

# Loop through the image list in chunks of 10
#for i in range(0, len(imgs), chunk_size):
#    image_chunk = imgs[i:i + chunk_size]
#    _showImages(True,imgs)

#from bigcrittercolor import writeColorMetrics


#import pandas as pd
#from bigcrittercolor.helpers.ids import _getIDsInFolder
# Load the CSV file
#file_path = 'D:/bcc/ringtails/records.csv'
#df = pd.read_csv(file_path)
#most_common_species = df['species'].mode()[0]
#ids = df[df['species'] == most_common_species]['img_id'].tolist()

#seg_ids = _getIDsInFolder("D:/bcc/ringtails/segments")
#ids = list(set(ids) & set(seg_ids))
#writeColorMetrics(img_ids=ids,data_folder="D:/bcc/ringtails",show_thresh_value=0.3)

createBCCDataFolder(parent_folder="D:/bcc",new_folder_name="all_beetles_test")