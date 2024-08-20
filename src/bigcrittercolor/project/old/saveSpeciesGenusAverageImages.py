import cv2
import numpy as np
import random
from bigcrittercolor.helpers.ids import _getRecordsColFromIDs
from bigcrittercolor.helpers import _getBCCIDs, _readBCCImgs, _bprint
from bigcrittercolor.helpers.image import _blackBgToTransparent

def saveSpeciesGenusAverageImages(data_folder, max_n_per_species=2, print_steps=True):
    ids = _getBCCIDs(type="segment", data_folder=data_folder)
    species_labels = _getRecordsColFromIDs(img_ids=ids, column="species", data_folder=data_folder)
    genus_labels = _getRecordsColFromIDs(img_ids=ids, column="genus", data_folder=data_folder)

    # Get unique species and genera
    unique_species = set(species_labels)
    unique_genera = set(genus_labels)

    # Dictionary to store species-level averages for each genus
    genus_species_averages = {genus: [] for genus in unique_genera}

    # Iterate through each unique species
    for species in unique_species:
        # Find matching IDs for the species
        matching_ids = [id for id, label in zip(ids, species_labels) if label == species]
        if len(matching_ids) > max_n_per_species:
            matching_ids = random.sample(matching_ids,max_n_per_species)
        # Read images
        imgs = _readBCCImgs(type="segment", img_ids=matching_ids, data_folder=data_folder)
        if not imgs:
            continue
        avg_img = getAverageImage(imgs)
        avg_img = _blackBgToTransparent(avg_img, threshold=0)

        # Save species-level average image
        species_underscore = species.replace(" ", "_")
        cv2.imwrite(data_folder + "/other/species_genus_average_images/" + species_underscore + ".png", avg_img)

        # Add species-level average to the corresponding genus
        genus = genus_labels[species_labels.index(species)]
        genus_species_averages[genus].append(avg_img)

    # Calculate genus-level averages based on species-level averages
    for genus, species_avgs in genus_species_averages.items():
        if species_avgs:
            genus_avg_img = getAverageImage(species_avgs)

            # Save genus-level average image
            genus_underscore = genus.replace(" ", "_")
            cv2.imwrite(data_folder + "/other/species_genus_average_images/" + genus_underscore + ".png", genus_avg_img)
def getAverageImage(image_list):
    if not image_list:
        raise ValueError("The image list is empty")

    def compute_average_dimensions(images):
        total_height = 0
        total_width = 0
        num_images = len(images)
        for img in images:
            h, w = img.shape[:2]
            total_height += h
            total_width += w
        avg_height = total_height // num_images
        avg_width = total_width // num_images
        return avg_height, avg_width

    def resize_images(images, target_height, target_width):
        resized_images = []
        for img in images:
            resized_img = cv2.resize(img, (target_width, target_height))
            resized_images.append(resized_img)
        return resized_images

    def compute_average_image(images):
        avg_image = np.mean(images, axis=0).astype(np.uint8)
        return avg_image

    avg_height, avg_width = compute_average_dimensions(image_list)
    resized_images = resize_images(image_list, avg_height, avg_width)
    avg_image = compute_average_image(resized_images)

    return avg_image

saveSpeciesGenusAverageImages(data_folder="D:/bcc/ringtails")