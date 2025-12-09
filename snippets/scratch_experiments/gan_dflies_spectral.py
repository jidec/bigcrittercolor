import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.nn.utils import spectral_norm
import pandas as pd
import os
import cv2

##################################################
# Conditional Generator
##################################################
class ConditionalGenerator(nn.Module):
    def __init__(self, encoding_dim, latent_dim, channels=3, feature_map_base=64, image_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoding_dim = encoding_dim
        self.channels = channels
        self.image_size = image_size

        # Project and reshape
        self.project = nn.Sequential(
            nn.Linear(latent_dim + encoding_dim, feature_map_base * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_base * 8 * 4 * 4),  # Consider InstanceNorm1d if issues persist
            nn.ReLU(True),
        )
        # Transposed conv stack
        self.main = nn.Sequential(
            nn.ConvTranspose2d(feature_map_base * 8, feature_map_base * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base * 4),  # Consider InstanceNorm2d if issues persist
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_base * 4, feature_map_base * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_base * 2, feature_map_base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_base, channels, 4, 2, 1, bias=False),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, noise, enc):
        x = torch.cat([noise, enc], dim=1)  # (batch, latent_dim + encoding_dim)
        x = self.project(x)
        x = x.view(x.size(0), -1, 4, 4)    # Reshape to (batch, feature_map_base*8, 4, 4)
        out = self.main(x)
        return out

##################################################
# Conditional Discriminator with Spectral Norm + Hinge
##################################################
class ConditionalDiscriminator(nn.Module):
    def __init__(self, encoding_dim, channels=3, feature_map_base=64, image_size=64):
        super().__init__()
        # Use spectral_norm on each layer
        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, feature_map_base, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_map_base, feature_map_base * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_map_base * 2, feature_map_base * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(feature_map_base * 4, feature_map_base * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Flatten and linear layer with spectral_norm
        in_features = feature_map_base * 8 * (image_size // 16) * (image_size // 16)  # e.g., 8192 for 64x64
        self.fc = spectral_norm(nn.Linear(in_features + encoding_dim, 1))

    def forward(self, img, enc):
        x_img = self.main(img)             # (batch, feature_map_base*8, 4, 4)
        x_img = x_img.view(x_img.size(0), -1)  # (batch, in_features)
        x = torch.cat([x_img, enc], dim=1)     # (batch, in_features + encoding_dim)
        out = self.fc(x)                        # (batch, 1)
        return out

##################################################
# Weight Initialization Function
##################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

##################################################
# Simple Dataset to Match Our Pipeline
##################################################
class EncImgDataset(Dataset):
    def __init__(self, encodings, images):
        super().__init__()
        if isinstance(encodings, np.ndarray):
            encodings = torch.from_numpy(encodings).float()
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        # Normalize encodings (L2 normalization)
        encodings = encodings / (encodings.norm(p=2, dim=1, keepdim=True) + 1e-8)

        self.encodings = encodings
        self.images = images

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        enc = self.encodings[idx]  # shape (encoding_dim,)
        img = self.images[idx]      # shape (C, H, W)
        return enc, img

##################################################
# Training Function with Hinge Loss and Gradient Clipping
##################################################
def trainFeaturesGAN(
        encodings_list,
        images_list,
        encoding_dim=2000,
        image_size=64,
        channels=3,
        latent_dim=100,
        batch_size=32,
        epochs=20,
        lr=2e-4,          # Updated to standard GAN learning rate
        beta1=0.5,        # Updated to standard GAN beta1
        n_critic=5,       # Number of critic updates per generator update
        save_interval=5,
        device="cuda",
        save_generator_path="generator_spectral_hinge.pth"
):
    # Convert lists to arrays
    encodings_arr = np.stack(encodings_list, axis=0)  # (N, encoding_dim)
    images_arr = np.stack(images_list, axis=0)        # (N, H, W, C) or (N, C, H, W)
    # If needed: (N,H,W,C)->(N,C,H,W)
    if images_arr.ndim == 4 and images_arr.shape[-1] in [1, 3]:
        images_arr = images_arr.transpose(0, 3, 1, 2)

    # Create dataset and dataloader
    dataset = EncImgDataset(encodings_arr, images_arr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Instantiate generator and discriminator
    netG = ConditionalGenerator(
        encoding_dim=encoding_dim,
        latent_dim=latent_dim,
        channels=channels,
        feature_map_base=64,
        image_size=image_size
    ).to(device)
    netG.apply(weights_init)  # Apply weight initialization

    netD = ConditionalDiscriminator(
        encoding_dim=encoding_dim,
        channels=channels,
        feature_map_base=64,
        image_size=image_size
    ).to(device)
    netD.apply(weights_init)  # Apply weight initialization

    # Optimizers with updated hyperparameters
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    # Training loop
    for epoch in range(epochs):
        netG.train()
        netD.train()

        for i, (enc_batch, real_imgs) in enumerate(dataloader):
            enc_batch = enc_batch.to(device)  # (batch, encoding_dim)
            real_imgs = real_imgs.to(device)  # (batch, C, H, W)
            current_bs = real_imgs.size(0)

            # =========================================================
            # Train Discriminator with Hinge Loss (n_critic times)
            # =========================================================
            for _ in range(n_critic):
                netD.zero_grad()

                # Real forward
                D_real = netD(real_imgs, enc_batch)  # shape (batch,1)
                lossD_real = F.relu(1.0 - D_real).mean()

                # Fake forward
                noise = torch.randn(current_bs, latent_dim, device=device)
                fake_imgs = netG(noise, enc_batch)
                D_fake = netD(fake_imgs.detach(), enc_batch)  # no grad to G here
                lossD_fake = F.relu(1.0 + D_fake).mean()

                lossD = lossD_real + lossD_fake
                lossD.backward()

                # Gradient Clipping (set higher max_norm)
                torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=5.0)

                optimizerD.step()

            # =========================================================
            # Train Generator with Hinge Loss
            # =========================================================
            netG.zero_grad()
            noise2 = torch.randn(current_bs, latent_dim, device=device)
            fake_imgs2 = netG(noise2, enc_batch)
            D_fake_for_G = netD(fake_imgs2, enc_batch)
            lossG = -D_fake_for_G.mean()

            lossG.backward()

            # Gradient Clipping (set higher max_norm)
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=5.0)

            optimizerG.step()

            # =========================================================
            # Monitoring and Debugging Outputs
            # =========================================================
            if i % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                    f"D Loss: {lossD.item():.4f}, G Loss: {lossG.item():.4f}"
                )
                # Print encoding statistics
                print(f"Encodings - min: {enc_batch.min().item():.4f}, max: {enc_batch.max().item():.4f}, "
                      f"mean: {enc_batch.mean().item():.4f}, std: {enc_batch.std().item():.4f}, "
                      f"NaN: {torch.isnan(enc_batch).any()}")

                # Print image statistics
                print(f"Real Images - min: {real_imgs.min().item():.4f}, max: {real_imgs.max().item():.4f}, "
                      f"mean: {real_imgs.mean().item():.4f}, NaN: {torch.isnan(real_imgs).any()}")

                # Print discriminator outputs
                D_real_mean = D_real.mean().item()
                D_fake_mean = D_fake.mean().item()
                print(f"D_real mean: {D_real_mean:.4f}, D_fake mean: {D_fake_mean:.4f}")

                # Check for NaNs in losses
                if torch.isnan(lossD) or torch.isnan(lossG):
                    print(f"NaN detected at Epoch {epoch + 1}, Batch {i}")
                    break

        # Save intervals
        if (epoch + 1) % save_interval == 0:
            torch.save(netG.state_dict(), f"generator_epoch_{epoch + 1}.pth")
            torch.save(netD.state_dict(), f"discriminator_epoch_{epoch + 1}.pth")

    # Save final generator
    torch.save(netG.state_dict(), save_generator_path)
    print(f"Training complete. Final generator saved to: {save_generator_path}")

##################################################
# Data Loading and Preprocessing
##################################################
csv_path = "D:/bcc/gan_test/bioencodings.csv"
df = pd.read_csv(csv_path)

print(df.columns)
print("Number of columns:", len(df.columns))
# Should be 2049 => 1 for image_name + 2048 for features

encodings_list = []
images_list = []

image_folder = "D:/bcc/gan_test/encodings_images"

for idx, row in df.iterrows():
    image_name = row["image_name"]  # e.g. "something.png"

    # Extract 2048 features
    # If they are columns: 0..2047
    feature_values = row.iloc[1:].values.astype(np.float32)  # shape (2048,)

    # Load image
    image_path = os.path.join(image_folder, image_name)
    if not os.path.isfile(image_path):
        print(f"Warning: Image file not found: {image_path}")
        continue

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"Warning: Could not load image: {image_path}")
        continue

    # Resize to 64x64 if needed
    img_bgr = cv2.resize(img_bgr, (64, 64))

    # Convert BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to float
    img_rgb = img_rgb.astype(np.float32)
    # Map [0..255] -> [-1..1]
    img_rgb = img_rgb / 127.5 - 1.0

    encodings_list.append(feature_values)
    images_list.append(img_rgb)

print("Loaded encodings:", len(encodings_list))
print("Loaded images:", len(images_list))

# Start Training
trainFeaturesGAN(
    encodings_list=encodings_list,
    images_list=images_list,
    encoding_dim=2048,   # <--- KEY
    image_size=64,
    channels=3,
    latent_dim=100,
    batch_size=16,
    epochs=5,
    lr=2e-4,             # Updated to standard learning rate
    beta1=0.5,           # Updated to standard beta1
    n_critic=5,          # Number of critic updates per generator update
    save_interval=1,
    device="cuda"        # or "cpu"
)

##################################################
# Image Generation Example
##################################################
device = "cuda"
index = 3
random_encoding = encodings_list[index]  # shape (2048,)

# Convert that encoding to a Torch tensor of shape (1, 2048)
random_encoding_t = torch.from_numpy(random_encoding).float().unsqueeze(0).to(device)

encoding_dim = 2048
latent_dim = 100
image_size = 64
channels = 3

# Prepare a single noise vector
noise = torch.randn(1, latent_dim, device=device)  # shape (1, 100)

# Re-create the generator architecture
generator = ConditionalGenerator(
    encoding_dim=encoding_dim,
    latent_dim=latent_dim,
    channels=channels,
    feature_map_base=64,
    image_size=image_size
).to(device)
generator.apply(weights_init)  # Apply weight initialization

# Load the saved weights
state_dict = torch.load("generator_final.pth", map_location=device)
generator.load_state_dict(state_dict)
generator.eval()

# Generate the image
with torch.no_grad():
    fake_image_t = generator(noise, random_encoding_t)  # shape (1, C, 64, 64)

# Convert from tensor to NumPy for OpenCV or other usage
# If your generator uses Tanh, values are in [-1,1]. Map them to [0..255].
fake_image_t = fake_image_t.squeeze(0).cpu()  # (C, H, W)
fake_image_np = (fake_image_t.numpy() + 1) / 2.0  # now in [0..1]
fake_image_np = (fake_image_np * 255.0).clip(0, 255).astype(np.uint8)  # (C, H, W)

# Transpose to (H, W, C) for OpenCV
fake_image_np = np.transpose(fake_image_np, (1, 2, 0))  # (64, 64, 3) in RGB

# Optional: convert RGB -> BGR if you plan to use cv2.imshow
fake_image_bgr = cv2.cvtColor(fake_image_np, cv2.COLOR_RGB2BGR)

# Now you can display or save it with OpenCV, e.g.
cv2.imshow("Fake Image", fake_image_bgr)
cv2.imshow("Real Image", (images_list[index] / 255.0).astype(np.float32))  # Ensure real image is in [0,1]
cv2.waitKey(0)
