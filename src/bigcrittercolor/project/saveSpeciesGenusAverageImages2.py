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

def saveSpeciesGenusAverageImages2(data_folder, max_n_per_species=2, print_steps=True):

    # assemble directory to train autoencoder
    shutil.rmtree(data_folder + "/other/species_mean_autoencoder_training")
    os.mkdir(data_folder + "/other/species_mean_autoencoder_training")
    os.mkdir(data_folder + "/other/species_mean_autoencoder_training/dummy_class")
    ids = _getBCCIDs(type="segment", data_folder=data_folder)
    species_labels = _getRecordsColFromIDs(img_ids=ids, column="species", data_folder=data_folder)
    genus_labels = _getRecordsColFromIDs(img_ids=ids, column="genus", data_folder=data_folder)
    # Get unique species and genera
    unique_species = set(species_labels)
    # Iterate through each unique species
    for species in unique_species:
        # Find matching IDs for the species
        matching_ids = [id for id, label in zip(ids, species_labels) if label == species]
        if len(matching_ids) > 20:
            matching_ids = random.sample(matching_ids, 20)
        # Read images
        imgs = _readBCCImgs(type="segment", img_ids=matching_ids, data_folder=data_folder)
        if not imgs:
            continue
        for img, id in zip(imgs, matching_ids):
            cv2.imwrite(data_folder + "/other/species_mean_autoencoder_training/dummy_class/" + id + ".png",img)

    # Define transformations and data loader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # train simple non-discriminative autoencoder
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
            # Forward pass
            output = autoencoder(img)
            loss = criterion(output, img)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # for each species, get latent representations, compute average, convert to image, and save
    # Iterate through each unique species
    for species in unique_species:
        # Find matching IDs for the species
        matching_ids = [id for id, label in zip(ids, species_labels) if label == species]
        if len(matching_ids) > 20:
            matching_ids = random.sample(matching_ids, 20)
        # Read images
        imgs = _readBCCImgs(type="segment", img_ids=matching_ids, data_folder=data_folder)
        imgs = resize_images_to_average_dimensions(imgs)

        # get encodings
        latent_representations = get_latent_representations(imgs, autoencoder)
        latent_avg = torch.mean(latent_representations, dim=0)
        avg_image_tensor = decode_latent(latent_avg, autoencoder).cpu()
        image = avg_image_tensor.permute(1, 2, 0).numpy()  # Change dimensions from (C, H, W) to (H, W, C)
        image = (image * 255).astype(np.uint8)  # Scale from [0, 1] to [0, 255]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #size = compute_average_dimensions(imgs)
        #image = cv2.resize(image, size)
        cv2.imwrite(data_folder + "/other/species_genus_average_images/" + species + ".png",image)

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
            # Convert the cv2 image (which is usually in BGR format) to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Convert the image to a tensor and add batch dimension
            img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1)  # Convert HWC to CHW
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

            # Normalize the tensor (assuming model requires normalization)
            img_tensor = img_tensor / 255.0  # Assuming normalization to [0, 1]

            # Move the tensor to the appropriate device
            img_tensor = img_tensor.to(device)

            # Pass through the model's encoder to get the latent representation
            latent = model.encoder(img_tensor)
            latent_representations.append(latent)

    return torch.cat(latent_representations, dim=0)

def decode_latent(latent_avg, model):
    model.eval()
    with torch.no_grad():
        avg_image = model.decoder(latent_avg.unsqueeze(0))
    return avg_image.squeeze()


def resize_images_to_average_dimensions(images):
    # Compute the average dimensions
    total_height = 0
    total_width = 0
    num_images = len(images)

    for img in images:
        h, w = img.shape[:2]
        total_height += h
        total_width += w

    avg_height = total_height // num_images
    avg_width = total_width // num_images

    # Resize images to the average dimensions
    resized_images = []

    for img in images:
        resized_img = cv2.resize(img, (avg_width, avg_height))
        resized_images.append(resized_img)

    return resized_images

#saveSpeciesGenusAverageImages2(data_folder="D:/bcc/ringtails")