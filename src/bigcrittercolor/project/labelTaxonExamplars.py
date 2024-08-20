import cv2
import pandas as pd
from bigcrittercolor.helpers import _readBCCImgs, _showImages, _getBCCIDs

def labelTaxonExemplars(img_ids=None, taxon="genus", data_folder=''):

    if img_ids is None:
        img_ids = _getBCCIDs(type="segment",data_folder=data_folder)

    # Load the records CSV file
    records_path = f"{data_folder}/records.csv"
    records_df = pd.read_csv(records_path)

    # Filter records by the provided img_ids
    filtered_records = records_df[records_df['img_id'].isin(img_ids)]
    #avg_size = calculate_average_size("D:/bcc/autoencoder/ringtails")
    # Dictionary to store the exemplar images
    exemplars = {}

    # Iterate through each unique species
    for species in filtered_records[taxon].unique():
        species_records = filtered_records[filtered_records[taxon] == species]
        species_img_ids = species_records['img_id'].tolist()

        # Use _readBCCImgs to get the list of images
        imgs = _readBCCImgs(species_img_ids,type="segment",sample_n=18,data_folder=data_folder)

        # Show images for the current species using _showImages
        _showImages(True,images=imgs,titles=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])

        # Take user input to identify one of the images as the exemplar
        exemplar_index = int(input(f"Select the exemplar image for taxon '{species}' by entering the image number: "))
        exemplar_img_id = species_img_ids[exemplar_index]

        # Save the selected exemplar img_id
        exemplars[species] = exemplar_img_id

    # Save the exemplars to a new CSV file
    exemplars_df = pd.DataFrame(list(exemplars.items()), columns=['taxon', 'exemplar_img_id'])
    exemplars_df.to_csv(f"{data_folder}/exemplars.csv", index=False)

    print("Exemplars have been saved to exemplars.csv")

#labelTaxonExemplars(taxon="genus",data_folder="D:/bcc/ringtails")