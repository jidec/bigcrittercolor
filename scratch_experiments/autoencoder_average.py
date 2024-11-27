import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import cv2
import numpy as np
import os
from PIL import Image

#bioencoder_model_explorer --config-path "D:/bcc/bioencoder_training/bioencoder_configs/explore_stage2.yml"
def calculate_average_size(folder_path):
    total_width = 0
    total_height = 0
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'tiff')):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    width, height = img.size
                    total_width += width
                    total_height += height
                    count += 1
    avg_width = total_width // count if count else 0
    avg_height = total_height // count if count else 0
    return avg_width, avg_height

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

# Define transformations and data loader
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

path = "D:/bcc/autoencoder"
dataset = datasets.ImageFolder(path, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
autoencoder = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        img = img.to(device)
        # Forward pass
        output = autoencoder(img)
        loss = criterion(output, img)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def get_latent_representations(dataloader, model):
    model.eval()
    latent_representations = []
    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            img = img.to(device)
            latent = model.encoder(img)
            latent_representations.append(latent)
    return torch.cat(latent_representations, dim=0)

def decode_latent(latent_avg, model):
    model.eval()
    with torch.no_grad():
        avg_image = model.decoder(latent_avg.unsqueeze(0))
    return avg_image.squeeze()

def show_image(tensor,size):
    tensor = tensor.cpu()
    image = tensor.permute(1, 2, 0).numpy()  # Change dimensions from (C, H, W) to (H, W, C)
    image = (image * 255).astype(np.uint8)  # Scale from [0, 1] to [0, 255]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image,size)
    cv2.imshow('Average Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Get latent representations and compute average
latent_representations = get_latent_representations(dataloader, autoencoder)
latent_avg = torch.mean(latent_representations, dim=0)

# Decode the average latent representation to an image
avg_image_tensor = decode_latent(latent_avg, autoencoder)

avg_size = calculate_average_size("D:/bcc/autoencoder/ringtails")
show_image(avg_image_tensor,avg_size)