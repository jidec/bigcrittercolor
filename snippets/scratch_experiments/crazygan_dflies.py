import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


##################################################
# Dataset
##################################################

class EncImgDataset(Dataset):
    """
    Holds encodings (N, encoding_dim=2048) and images (N, C, H, W).
    """
    def __init__(self, encodings, images, transform=None):
        super().__init__()
        if isinstance(encodings, np.ndarray):
            encodings = torch.from_numpy(encodings).float()
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        # Normalize encodings (L2 normalization)
        encodings = encodings / encodings.norm(p=2, dim=1, keepdim=True)

        self.encodings = encodings
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        enc = self.encodings[idx]  # shape (2048,)
        img = self.images[idx]     # shape (C, H, W)
        if self.transform:
            img = self.transform(img)
        return enc, img


##################################################
# Generator: conditional on 2048-dim enc
##################################################

class ConditionalGenerator(nn.Module):
    def __init__(self, encoding_dim=2048, latent_dim=100,
                 channels=3, feature_map_base=64, image_size=64):
        """
        A typical DCGAN-style generator, but with (noise + encoding_dim).
        """
        super(ConditionalGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.encoding_dim = encoding_dim
        self.channels = channels
        self.image_size = image_size

        # Project + reshape
        # Input is (latent_dim + encoding_dim) => e.g. 100 + 2048 = 2148
        self.project = nn.Sequential(
            nn.Linear(latent_dim + encoding_dim, feature_map_base * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_base * 8 * 4 * 4),
            nn.ReLU(True),
        )

        # Transposed conv to upsample 4x4 -> 64x64
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
        """
        noise: (batch_size, latent_dim)
        enc:   (batch_size, 2048)
        """
        x = torch.cat([noise, enc], dim=1)  # shape (batch_size, latent_dim+2048)
        x = self.project(x)
        # shape => (batch_size, 64*8, 4, 4) = (batch_size, 512, 4, 4)
        x = x.view(x.size(0), -1, 4, 4)
        out = self.main(x)
        return out


##################################################
# Discriminator: conditional on 2048-dim enc
##################################################

class ConditionalDiscriminator(nn.Module):
    def __init__(self, encoding_dim=2048, channels=3,
                 feature_map_base=64, image_size=64):
        """
        Typical DCGAN-style discriminator that merges image features + 2048-dim enc.
        """
        super(ConditionalDiscriminator, self).__init__()

        # Downsample image 64x64 -> 4x4
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
            nn.LeakyReLU(0.2, inplace=True)
        )

        # For a 64x64 input, after 4 downsamplings you get (feature_map_base*8, 4, 4) => 512 * 4 * 4 = 8192
        # Add 2048 from enc => total is 8192 + 2048 = 10240
        in_features = feature_map_base * 8 * (image_size // 16) * (image_size // 16)  # 8192
        self.fc = nn.Sequential(
            nn.Linear(in_features + encoding_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            # We'll use BCEWithLogitsLoss, so no final sigmoid here
        )

    def forward(self, img, enc):
        """
        img: (batch_size, channels, 64, 64)
        enc: (batch_size, 2048)
        """
        x_img = self.main(img)              # (batch_size, 512, 4, 4)
        x_img = x_img.view(x_img.size(0), -1)  # (batch_size, 8192)

        x = torch.cat([x_img, enc], dim=1)  # (batch_size, 8192+2048=10240)
        out = self.fc(x)                    # (batch_size, 1)
        return out


##################################################
# Main Training Function
##################################################

def trainFeaturesGAN(
    encodings_list,
    images_list,
    encoding_dim=2048,  # Now 2048
    image_size=64,
    channels=3,
    latent_dim=100,
    batch_size=32,
    epochs=50,
    lr=2e-4,
    beta1=0.5,
    save_interval=10,
    device="cuda",
    save_generator_path="generator_final.pth"
):
    """
    Train a conditional DCGAN with 2048-dim encodings.
    """
    # 1) Convert lists to arrays
    encodings_arr = np.stack(encodings_list, axis=0)  # shape (N, 2048)
    images_arr = np.stack(images_list, axis=0)        # shape (N, H, W, C) or (N, C, H, W)

    # If images are (N, H, W, C), transpose to (N, C, H, W)
    if images_arr.ndim == 4 and images_arr.shape[-1] in [1, 3]:
        images_arr = images_arr.transpose(0, 3, 1, 2)  # (N, C, 64, 64)

    # 2) Make Dataset / DataLoader
    dataset = EncImgDataset(encodings_arr, images_arr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3) Instantiate models (Generator & Discriminator)
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

    # 4) Loss & Optimizers
    criterion = nn.BCEWithLogitsLoss()
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

    real_label = 1.0
    fake_label = 0.0

    # (Optional) fixed noise for debugging/generating samples
    fixed_noise = torch.randn(batch_size, latent_dim, device=device)
    fixed_enc = None

    # 5) Training loop
    for epoch in range(epochs):
        netG.train()
        netD.train()

        for i, (enc_batch, real_imgs) in enumerate(dataloader):
            #print("Real img stats:", real_imgs.min().item(), real_imgs.max().item())

            enc_batch = enc_batch.to(device)    # (batch_size, 2048)
            real_imgs = real_imgs.to(device)    # (batch_size, C, 64, 64)
            current_bs = real_imgs.size(0)

            # ---- Train Discriminator ----
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
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)

            optimizerD.step()

            # ---- Train Generator ----
            netG.zero_grad()
            label_gen = torch.full((current_bs, 1), real_label, device=device)
            output_gen = netD(fake_imgs, enc_batch)
            lossG = criterion(output_gen, label_gen)
            lossG.backward()
            torch.nn.utils.clip_grad_norm_(netG.parameters(), max_norm=1.0)

            optimizerG.step()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] "
                    f"D Loss: {lossD.item():.4f}, G Loss: {lossG.item():.4f}"
                )

        # (Optional) Save intermediate
        if (epoch + 1) % save_interval == 0:
            torch.save(netG.state_dict(), f"generator_epoch_{epoch+1}.pth")
            torch.save(netD.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

    # 6) Save final generator
    torch.save(netG.state_dict(), save_generator_path)
    print("Training complete. Final generator saved to:", save_generator_path)

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
    # Optionally, map [0..255] -> [-1..1] if your generator uses Tanh
    # img_rgb = img_rgb / 127.5 - 1.0

    encodings_list.append(feature_values)
    images_list.append(img_rgb)

print("Loaded encodings:", len(encodings_list))
print("Loaded images:", len(images_list))

trainFeaturesGAN(
    encodings_list=encodings_list,
    images_list=images_list,
    encoding_dim=2048,   # <--- KEY
    image_size=64,
    channels=3,
    latent_dim=100,
    batch_size=8,
    epochs=5,
    lr=5e-5,
    beta1=0.5,
    save_interval=1,
    device="cuda"        # or "cpu"
)

device = "cuda"
index = 3
random_encoding = encodings_list[index]  # shape (2048,)

# 2) Convert that encoding to a Torch tensor of shape (1, 2048)
random_encoding_t = torch.from_numpy(random_encoding).float().unsqueeze(0).to(device)

encoding_dim = 2048
latent_dim = 100
image_size = 64
channels = 3

# 3) Prepare a single noise vector
noise = torch.randn(1, latent_dim, device=device)  # shape (1, 100)

# 4) Re-create the generator architecture
generator = ConditionalGenerator(
    encoding_dim=encoding_dim,
    latent_dim=latent_dim,
    channels=channels,
    feature_map_base=64,
    image_size=image_size
).to(device)

# 5) Load the saved weights
state_dict = torch.load("generator_final.pth", map_location=device)
generator.load_state_dict(state_dict)
generator.eval()

# 6) Generate the image
with torch.no_grad():
    fake_image_t = generator(noise, random_encoding_t)  # shape (1, C, 64, 64)

# 7) Convert from tensor to NumPy for OpenCV or other usage
#    If your generator uses Tanh, values are in [-1,1]. Map them to [0..255].
fake_image_t = fake_image_t.squeeze(0).cpu()  # (C, H, W)
fake_image_np = (fake_image_t.numpy() + 1) / 2.0  # now in [0..1]
fake_image_np = (fake_image_np * 255.0).clip(0, 255).astype(np.uint8)  # (C, H, W)

# 8) Transpose to (H, W, C) for OpenCV
fake_image_np = np.transpose(fake_image_np, (1, 2, 0))  # (64, 64, 3) in RGB

# Optional: convert RGB -> BGR if you plan to use cv2.imshow
fake_image_bgr = cv2.cvtColor(fake_image_np, cv2.COLOR_RGB2BGR)

# Now you can display or save it with OpenCV, e.g.
cv2.imshow("Fake Image", fake_image_bgr)
cv2.imshow("Real Image", images_list[index] / 255)
cv2.waitKey(0)