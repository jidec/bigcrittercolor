import os
import pandas as pd
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from bigcrittercolor.helpers import _readBCCImgs, _bprint
from bigcrittercolor.helpers.image import _blackBgToTransparent

def saveBioencoderSpeciesGenusExemplars(n_pcs=8,data_folder=''):

    data_path = os.path.join(data_folder, "other", "bioencodings.csv")
    df = pd.read_csv(data_path)
    df['img_id'] = df['image_name'].str.replace('.png', '', regex=False)

    # Load the species and genus data
    records_path = os.path.join(data_folder, "records.csv")
    records_df = pd.read_csv(records_path)
    df = df.merge(records_df[['img_id', 'species', 'genus']], on='img_id')

    x_columns = [col for col in df.columns if any(char.isdigit() for char in col)]
    df[x_columns] = df[x_columns].fillna(df[x_columns].mean())
    pca = PCA(n_components=20)
    pca_scores = pca.fit_transform(df[x_columns])

    # Add the PCA scores to the original dataframe
    pc_columns = [f'PC{i + 1}' for i in range(n_pcs)]
    pca_df = pd.DataFrame(pca_scores, columns=pc_columns)
    df = pd.concat([df, pca_df], axis=1)

    # Calculate the mean PC values for each species
    species_means = df.groupby('species')[pc_columns].mean()

    # Find the img_id with PC values closest to the mean of the species
    species_exemplars = []
    for species, group in df.groupby('species'):
        mean_values = species_means.loc[species].values.reshape(1, -1)
        distances = euclidean_distances(group[pc_columns], mean_values)
        closest_idx = distances.argmin()
        closest_img_id = group.iloc[closest_idx]['img_id']
        species_exemplars.append((species, closest_img_id))

    # Save the closest image for each species
    output_folder = os.path.join(data_folder, "other", "species_genus_exemplars")
    os.makedirs(output_folder, exist_ok=True)

    for species, img_id in species_exemplars:
        img = _readBCCImgs(img_id, type="segment", data_folder=data_folder)
        img = _blackBgToTransparent(img)
        img_path = os.path.join(output_folder, f"{species}.png")
        cv2.imwrite(img_path, img)

    # Calculate the mean PC values for each genus
    genus_means = df.groupby('genus')[pc_columns].mean()

    # Find the img_id with PC values closest to the mean of the genus
    genus_exemplars = []
    for genus, group in df.groupby('genus'):
        mean_values = genus_means.loc[genus].values.reshape(1, -1)
        distances = euclidean_distances(group[pc_columns], mean_values)
        closest_idx = distances.argmin()
        closest_img_id = group.iloc[closest_idx]['img_id']
        genus_exemplars.append((genus, closest_img_id))

    # Save the closest image for each genus
    for genus, img_id in genus_exemplars:
        img = _readBCCImgs(img_id, type="segment", data_folder=data_folder)
        img = _blackBgToTransparent(img)
        img_path = os.path.join(output_folder, f"{genus}.png")
        cv2.imwrite(img_path, img)

#saveBioencoderSpeciesGenusExemplars(data_folder="D:/bcc/ringtails")