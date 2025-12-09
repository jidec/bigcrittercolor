import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import pandas as pd
import os

class ConditionalGenerator(nn.Module):
    def __init__(self, encoding_dim=2048, latent_dim=100,
                 channels=3, feature_map_base=64, image_size=64):
        """
        Generator in a conditional WGAN-GP setup.
        Takes (noise + encoding) -> outputs an image (channels, image_size, image_size).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.encoding_dim = encoding_dim
        self.channels = channels
        self.image_size = image_size

        self.project = nn.Sequential(
            nn.Linear(latent_dim + encoding_dim, feature_map_base * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_base * 8 * 4 * 4),
            nn.ReLU(True),
        )

        self.main = nn.Sequential(
            nn.ConvTranspose2d(feature_map_base*8, feature_map_base*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_base*4, feature_map_base*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_base*2, feature_map_base, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_base),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_map_base, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, enc):
        # noise: (batch_size, latent_dim)
        # enc:   (batch_size, encoding_dim)
        x = torch.cat([noise, enc], dim=1)  # (batch, latent_dim+encoding_dim)
        x = self.project(x)
        x = x.view(x.size(0), -1, 4, 4)     # shape: (batch, 512, 4, 4) if base=64
        out = self.main(x)
        return out

class ConditionalDiscriminator(nn.Module):
    def __init__(self, encoding_dim=2048, channels=3,
                 feature_map_base=64, image_size=64):
        """
        Critic in WGAN-GP. Outputs a single scalar (no sigmoid).
        Condition on image + encoding.
        """
        super().__init__()
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
        # flatten then combine with encoding
        in_features = feature_map_base * 8 * (image_size // 16) * (image_size // 16)  # e.g. 8192 if 64x64
        self.fc = nn.Linear(in_features + encoding_dim, 1)  # raw scalar

    def forward(self, img, enc):
        # img: (batch, channels, H, W)
        # enc: (batch, encoding_dim)
        x_img = self.main(img)
        x_img = x_img.view(x_img.size(0), -1)  # flatten -> (batch, in_features)
        x = torch.cat([x_img, enc], dim=1)     # (batch, in_features+encoding_dim)
        out = self.fc(x)
        return out  # shape (batch, 1) (no sigmoid!)

def gradient_penalty(critic, real_imgs, fake_imgs, real_enc, fake_enc, device="cuda", gp_lambda=10.0):
    """
    Compute gradient penalty for conditional WGAN-GP.
    Interpolate both images + encodings, then compute the gradient wrt the critic's output.
    critic: the ConditionalDiscriminator
    real_imgs, fake_imgs: (batch, C, H, W)
    real_enc, fake_enc:   (batch, encoding_dim)
    gp_lambda: weighting factor (often 10)
    """
    batch_size = real_imgs.size(0)

    # Sample random alphas for images
    alpha_img = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha_img = alpha_img.expand_as(real_imgs)  # shape (batch, C, H, W)

    # Interpolate images
    interpolated_imgs = alpha_img * real_imgs + (1 - alpha_img) * fake_imgs

    # Similarly for encodings
    alpha_enc = torch.rand(batch_size, 1, device=device)
    alpha_enc = alpha_enc.expand_as(real_enc)  # shape (batch, encoding_dim)

    interpolated_enc = alpha_enc * real_enc + (1 - alpha_enc) * fake_enc

    # Make them require grad
    interpolated_imgs.requires_grad_(True)
    interpolated_enc.requires_grad_(True)

    # Forward pass
    crit_out = critic(interpolated_imgs, interpolated_enc)  # shape (batch, 1)

    # Compute gradients wrt the interpolated inputs
    grad_outputs = torch.ones_like(crit_out, device=device)
    grads = torch.autograd.grad(
        outputs=crit_out,
        inputs=[interpolated_imgs, interpolated_enc],
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )

    # grads is a tuple: (grad_imgs, grad_enc)
    grad_imgs = grads[0].view(batch_size, -1)  # (batch, C*H*W)
    grad_encs = grads[1].view(batch_size, -1)  # (batch, encoding_dim)
    combined_grads = torch.cat([grad_imgs, grad_encs], dim=1)  # (batch, C*H*W + encoding_dim)

    grad_norm = combined_grads.norm(2, dim=1)  # L2 norm per sample, shape (batch,)

    # Penalty = mean( (||grad||2 - 1)^2 )
    gp = gp_lambda * ((grad_norm - 1) ** 2).mean()
    return gp

class EncImgDataset(Dataset):
    def __init__(self, encodings, images):
        super().__init__()
        if isinstance(encodings, np.ndarray):
            encodings = torch.from_numpy(encodings).float()
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images).float()

        # Normalize encodings (L2 normalization)
        encodings = encodings / encodings.norm(p=2, dim=1, keepdim=True)

        self.encodings = encodings
        self.images = images

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        enc = self.encodings[idx]  # shape (encoding_dim,)
        img = self.images[idx]     # shape (C, H, W)
        return enc, img


def trainFeaturesGAN(
    encodings_list,
    images_list,
    encoding_dim=2048,
    image_size=64,
    channels=3,
    latent_dim=100,
    batch_size=32,
    epochs=5,
    lr=1e-4,
    beta1=0.5,
    beta2=0.9,
    gp_lambda=10.0,
    device="cuda",
    save_interval=2,
    save_generator_path="generator_wgangp.pth"
):
    """
    Train a conditional WGAN-GP on (encodings, images).
    """
    # 1) Convert lists -> arrays
    encodings_arr = np.stack(encodings_list, axis=0)  # (N, encoding_dim)
    images_arr = np.stack(images_list, axis=0)        # (N, C, H, W) or (N, H, W, C)
    if images_arr.ndim == 4 and images_arr.shape[-1] in [1,3]:
        images_arr = images_arr.transpose(0, 3, 1, 2)

    dataset = EncImgDataset(encodings_arr, images_arr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2) Build generator and discriminator (critic)
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

    # 3) Optimizers
    # WGAN-GP typically uses Adam with lower betas, e.g. beta1=0, beta2=0.9 or 0.99
    # but you can experiment. We'll do (beta1=0.5, beta2=0.9) here
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

    # 4) Training loop
    for epoch in range(epochs):
        netG.train()
        netD.train()

        for i, (enc_batch, real_imgs) in enumerate(dataloader):
            current_bs = real_imgs.size(0)
            enc_batch = enc_batch.to(device)  # (batch, encoding_dim)
            real_imgs = real_imgs.to(device)  # (batch, C, H, W)

            # ---------------------
            # Train Critic (netD)
            # ---------------------
            optimizerD.zero_grad()

            # real forward
            D_real = netD(real_imgs, enc_batch).mean()  # average over batch

            # fake forward
            noise = torch.randn(current_bs, latent_dim, device=device)
            fake_imgs = netG(noise, enc_batch)
            D_fake = netD(fake_imgs.detach(), enc_batch).mean()

            # gradient penalty
            gp = gradient_penalty(netD, real_imgs, fake_imgs, enc_batch, enc_batch, device=device, gp_lambda=gp_lambda)

            # WGAN-GP critic loss
            lossD = D_fake - D_real + gp
            lossD.backward()
            torch.nn.utils.clip_grad_norm_(netD.parameters(), max_norm=1.0)
            optimizerD.step()

            # ---------------------
            # Train Generator (netG)
            # ---------------------
            optimizerG.zero_grad()

            # forward again (new noise if you prefer, or same)
            noise2 = torch.randn(current_bs, latent_dim, device=device)
            fake_imgs2 = netG(noise2, enc_batch)
            D_fake_for_G = netD(fake_imgs2, enc_batch).mean()
            # generator wants to maximize D_fake => minimize -D_fake
            lossG = -D_fake_for_G
            lossG.backward()
            optimizerG.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] ",
                      f"D Loss: {lossD.item():.4f}, G Loss: {lossG.item():.4f}, GP: {gp.item():.4f}")

        # save intervals
        if (epoch + 1) % save_interval == 0:
            torch.save(netG.state_dict(), f"generator_epoch_{epoch+1}.pth")
            torch.save(netD.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

    # save final generator
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
    batch_size=4,
    epochs=5,
    lr=5e-5,
    beta1=0,
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

import cv2

# Optional: convert RGB -> BGR if you plan to use cv2.imshow
fake_image_bgr = cv2.cvtColor(fake_image_np, cv2.COLOR_RGB2BGR)

# Now you can display or save it with OpenCV, e.g.
cv2.imshow("Fake Image", fake_image_bgr)
cv2.imshow("Real Image", images_list[index] / 255)
cv2.waitKey(0)