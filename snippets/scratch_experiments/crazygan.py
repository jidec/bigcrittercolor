import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


##################################################
# Dataset (unchanged)
##################################################
class EncImgDataset(Dataset):
    def __init__(self, encodings, images, transform=None):
        super().__init__()
        if isinstance(encodings, np.ndarray):
            encodings = torch.from_numpy(encodings).float()
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()
        self.encodings = encodings
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        enc = self.encodings[idx]
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return enc, img


##################################################
# Generator
##################################################
class ConditionalGenerator(nn.Module):
    def __init__(self, encoding_dim, latent_dim, channels=3, feature_map_base=64, image_size=64):
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.encoding_dim = encoding_dim
        self.channels = channels
        self.image_size = image_size

        # Project + reshape
        self.project = nn.Sequential(
            nn.Linear(latent_dim + encoding_dim, feature_map_base * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_base * 8 * 4 * 4),
            nn.ReLU(True),
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(feature_map_base * 8, feature_map_base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_base * 4, feature_map_base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_base * 2, feature_map_base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_base, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, enc):
        x = torch.cat([noise, enc], dim=1)  # (batch_size, latent_dim + encoding_dim)
        x = self.project(x)
        x = x.view(x.size(0), -1, 4, 4)
        out = self.main(x)
        return out


##################################################
# Discriminator (unchanged)
##################################################
class ConditionalDiscriminator(nn.Module):
    def __init__(self, encoding_dim, channels=3, feature_map_base=64, image_size=64):
        super(ConditionalDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, feature_map_base, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_base, feature_map_base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_base * 2, feature_map_base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_map_base * 4, feature_map_base * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        in_features = feature_map_base * 8 * (image_size // 16) * (image_size // 16)
        self.fc = nn.Sequential(
            nn.Linear(in_features + encoding_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            # We'll typically use BCEWithLogitsLoss so no sigmoid here
        )

    def forward(self, img, enc):
        x_img = self.main(img)
        x_img = x_img.view(x_img.size(0), -1)
        x = torch.cat([x_img, enc], dim=1)
        out = self.fc(x)
        return out


##################################################
# 3) The trainFeaturesGAN function
##################################################
def trainFeaturesGAN(
    encodings_list,
    images_list,
    encoding_dim=2000,
    image_size=64,
    channels=3,
    latent_dim=100,
    batch_size=32,
    epochs=50,
    lr=2e-4,
    beta1=0.5,
    save_interval=10,
    device="cuda",
    # New param: path where we'll save the generator weights at the end
    save_generator_path="generator_final.pth"
):
    # ---------------------------
    # Step 0: Convert lists to arrays
    # ---------------------------
    encodings_arr = np.stack(encodings_list, axis=0)  # shape (N, encoding_dim)
    images_arr = np.stack(images_list, axis=0)        # shape (N, H, W, C) or (N, C, H, W)

    # If needed: convert (N, H, W, C) -> (N, C, H, W)
    if images_arr.ndim == 4 and images_arr.shape[-1] in [1,3]:
        images_arr = images_arr.transpose(0, 3, 1, 2)

    # ---------------------------
    # Step 1: Dataset / DataLoader
    # ---------------------------
    dataset = EncImgDataset(encodings_arr, images_arr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # ---------------------------
    # Step 2: Instantiate models
    # ---------------------------
    netG = ConditionalGenerator(
        encoding_dim=encoding_dim,
        latent_dim=latent_dim,
        channels=channels,
        feature_map_base=64,
        image_size=image_size
    ).to(device)

    netD = ConditionalDiscriminator(
        encoding_dim=encoding_dim,
        channels=channels,
        feature_map_base=64,
        image_size=image_size
    ).to(device)

    # ---------------------------
    # Step 3: Loss / Optimizers
    # ---------------------------
    criterion = nn.BCEWithLogitsLoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0

    # Optional: fixed noise for sample generation
    fixed_noise = torch.randn(batch_size, latent_dim, device=device)
    fixed_enc = None

    # ---------------------------
    # Step 4: Training Loop
    # ---------------------------
    for epoch in range(epochs):
        netG.train()
        netD.train()

        for i, (enc_batch, real_imgs) in enumerate(dataloader):
            enc_batch = enc_batch.to(device)
            real_imgs = real_imgs.to(device)
            current_bs = real_imgs.size(0)

            # Train Discriminator
            netD.zero_grad()
            label_real = torch.full((current_bs, 1), real_label, device=device)
            output_real = netD(real_imgs, enc_batch)
            lossD_real = criterion(output_real, label_real)

            noise = torch.randn(current_bs, latent_dim, device=device)
            fake_imgs = netG(noise, enc_batch)
            label_fake = torch.full((current_bs, 1), fake_label, device=device)
            output_fake = netD(fake_imgs.detach(), enc_batch)
            lossD_fake = criterion(output_fake, label_fake)

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label_gen = torch.full((current_bs, 1), real_label, device=device)
            output_gen = netD(fake_imgs, enc_batch)
            lossG = criterion(output_gen, label_gen)
            lossG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                    f"D Loss: {lossD.item():.4f}, G Loss: {lossG.item():.4f}"
                )

        # Save intervals (optional)
        if (epoch + 1) % save_interval == 0:
            torch.save(netG.state_dict(), f"generator_epoch_{epoch+1}.pth")
            torch.save(netD.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

    # Save final generator weights
    torch.save(netG.state_dict(), save_generator_path)
    print("Training complete. Final generator saved to:", save_generator_path)


num_samples = 100       # how many examples we want
encoding_dim = 2000     # the length of each encoding vector
image_size = 64         # image height/width
channels = 3            # e.g. RGB
latent_dim = 100        # dimension of the random noise input to the generator

# Generate random encodings_list:
# shape will be [num_samples], each element shape=(encoding_dim,)
encodings_list = [
    np.random.randn(encoding_dim).astype(np.float32)
    for _ in range(num_samples)
]

# Generate random images_list:
# shape will be [num_samples], each element shape=(image_size, image_size, channels)
images_list = [
    np.random.randn(image_size, image_size, channels).astype(np.float32)
    for _ in range(num_samples)
]

fixed_images = []
for img in images_list:  # each img: shape (H, W, C)
    # transpose to shape (C, H, W)
    img_chw = np.transpose(img, (2, 0, 1))
    fixed_images.append(img_chw)
images_list = fixed_images  # now each entry is (C, H, W)

# Now call the trainFeaturesGAN function you defined
trainFeaturesGAN(
    encodings_list=encodings_list,   # the list of random encodings
    images_list=images_list,         # the list of random images
    encoding_dim=encoding_dim,
    image_size=image_size,
    channels=channels,
    latent_dim=latent_dim,           # the noise dimension
    batch_size=8,                    # small batch size for a quick test
    epochs=2,                        # just a couple epochs for demonstration
    lr=2e-4,
    beta1=0.5,
    save_interval=1,                # save every epoch in this demo
    device="cpu"                    # run on CPU for simplicity (if no GPU)
)

def generateImagesFromFeatures(
        model_path,
        encodings,
        encoding_dim=2000,
        latent_dim=100,
        image_size=64,
        channels=3,
        device="cuda",
        noise=None
):
    """
    Load a saved generator, then generate images for the given encodings.

    Args:
        model_path (str): Path to the saved generator weights (.pth).
        encodings (np.ndarray or torch.Tensor):
            - shape (encoding_dim,) for a single encoding or
            - shape (N, encoding_dim) for a batch
        encoding_dim (int): Must match the dimension used in training.
        latent_dim (int): Must match the dimension used in training.
        image_size (int): Must match the dimension used in training.
        channels (int): Must match the dimension used in training.
        device (str): "cuda" or "cpu".
        noise (torch.Tensor, optional): shape (N, latent_dim).
            If None, we generate new random noise. If provided, we use that.

    Returns:
        torch.Tensor: Generated images in shape:
            - (C, H, W) if input is a single encoding or
            - (N, C, H, W) if a batch of encodings
        The values will be in [-1, 1] if using Tanh in the generator.
    """
    # 1) Recreate the same generator architecture
    generator = ConditionalGenerator(
        encoding_dim=encoding_dim,
        latent_dim=latent_dim,
        channels=channels,
        feature_map_base=64,  # must match training
        image_size=image_size  # must match training
    )

    # 2) Load weights
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint)
    generator.to(device)
    generator.eval()

    # 3) Convert encodings to torch if needed
    if isinstance(encodings, np.ndarray):
        encodings = torch.from_numpy(encodings).float()
    if encodings.ndim == 1:
        # Single encoding -> shape (1, encoding_dim)
        encodings = encodings.unsqueeze(0)

    encodings = encodings.to(device)
    batch_size = encodings.size(0)

    # 4) Prepare noise
    if noise is None:
        noise = torch.randn(batch_size, latent_dim, device=device)

    # 5) Generate images
    with torch.no_grad():
        fake_images = generator(noise, encodings)  # (N, C, H, W)

    # 6) If single encoding, squeeze out batch dimension
    if batch_size == 1:
        fake_images = fake_images.squeeze(0)  # (C, H, W)

    # Move images back to CPU for further processing
    fake_images = fake_images.cpu()

    return fake_images

# Let's assume you used the default saving path "generator_final.pth"
# and that your training used these same parameters:
model_path = "generator_final.pth"
# Single encoding
one_enc = np.random.randn(2000).astype(np.float32)  # (encoding_dim=2000)

# Generate
gen_img = generateImagesFromFeatures(
    model_path=model_path,
    encodings=one_enc,
    encoding_dim=2000,
    latent_dim=100,
    image_size=64,
    channels=3,
    device="cpu"
)
print("Generated image shape:", gen_img.shape)
# -> (C, 64, 64) if single encoding

import cv2

# Move [-1,1] -> [0,1]
img_np = (gen_img.numpy() + 1.0) / 2.0
# Move from (C,H,W) -> (H,W,C)
img_np = np.transpose(img_np, (1, 2, 0))
# Scale up to [0..255]
img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)

cv2.imshow("0",img_np)
cv2.waitKey(0)