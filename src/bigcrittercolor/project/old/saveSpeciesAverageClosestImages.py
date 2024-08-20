import cv2
import numpy as np
import random
from bigcrittercolor.helpers.ids import _getRecordsColFromIDs
from bigcrittercolor.helpers import _getBCCIDs, _readBCCImgs, _bprint
from bigcrittercolor.helpers.image import _blackBgToTransparent

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from PIL import Image
import shutil

def saveSpeciesAverageClosestImages(data_folder, max_n_per_species=2, print_steps=True):
    # Assemble directory to train autoencoder
    if os.path.exists(data_folder + "/other/species_mean_autoencoder_training"):
        shutil.rmtree(data_folder + "/other/species_mean_autoencoder_training")
    os.mkdir(data_folder + "/other/species_mean_autoencoder_training")
    os.mkdir(data_folder + "/other/species_mean_autoencoder_training/dummy_class")
    ids = _getBCCIDs(type="segment", data_folder=data_folder)
    species_labels = _getRecordsColFromIDs(img_ids=ids, column="species", data_folder=data_folder)
    genus_labels = _getRecordsColFromIDs(img_ids=ids, column="genus", data_folder=data_folder)
    unique_species = set(species_labels)

    # Train autoencoder
    for species in unique_species:
        matching_ids = [id for id, label in zip(ids, species_labels) if label == species]
        if len(matching_ids) > 20:
            matching_ids = random.sample(matching_ids, 20)
        imgs = _readBCCImgs(type="segment", img_ids=matching_ids, data_folder=data_folder)
        if not imgs:
            continue
        for img, id in zip(imgs, matching_ids):
            cv2.imwrite(data_folder + "/other/species_mean_autoencoder_training/dummy_class/" + id + ".png", img)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    path = data_folder + "/other/species_mean_autoencoder_training"
    dataset = datasets.ImageFolder(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    autoencoder = Autoencoder().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    num_epochs = 100

    for epoch in range(num_epochs):
        for data in dataloader:
            img, _ = data
            img = img.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            output = autoencoder(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Find image with encoding closest to the mean encoding
    for species in unique_species:
        print('wrote')
        matching_ids = [id for id, label in zip(ids, species_labels) if label == species]
        if len(matching_ids) > 20:
            matching_ids = random.sample(matching_ids, 20)
        imgs = _readBCCImgs(type="segment", img_ids=matching_ids, data_folder=data_folder)
        imgs = resize_images_to_average_dimensions(imgs)

        latent_representations = get_latent_representations(imgs, autoencoder)
        latent_avg = torch.mean(latent_representations, dim=0)

        # Calculate the distance of each latent representation to the mean
        distances = torch.norm(latent_representations - latent_avg, dim=1)
        closest_idx = torch.argmin(distances)

        # Save the image corresponding to the closest latent representation
        closest_image = imgs[closest_idx]
        cv2.imwrite(data_folder + "/other/species_genus_average_images/" + species + ".png", closest_image)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # [32, 64, 64]
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # [64, 32, 32]
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # [32, 64, 64]
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # [3, 128, 128]
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def get_latent_representations(image_list, model):
    model.eval()
    latent_representations = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with torch.no_grad():
        for img in image_list:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor / 255.0
            img_tensor = img_tensor.to(device)
            latent = model.encoder(img_tensor)
            latent_representations.append(latent)

    return torch.cat(latent_representations, dim=0)

def resize_images_to_average_dimensions(images):
    total_height = sum(img.shape[0] for img in images)
    total_width = sum(img.shape[1] for img in images)
    avg_height = total_height // len(images)
    avg_width = total_width // len(images)

    resized_images = [cv2.resize(img, (avg_width, avg_height)) for img in images]
    return resized_images

# Example usage
saveSpeciesAverageClosestImages(data_folder="D:/bcc/ringtails")