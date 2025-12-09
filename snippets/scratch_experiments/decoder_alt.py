import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import cv2
from torch.cuda import amp
from torchvision.models import vgg16

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchmetrics.functional import multiscale_structural_similarity_index_measure as ms_ssim_metric

class MultiScaleSSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, reduction='mean'):
        """
        Initializes the MultiScaleSSIMLoss.

        Args:
            data_range (float): The range of the input images (usually 1.0 if images are in [0, 1]).
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'mean' will average the loss over the batch.
        """
        super(MultiScaleSSIMLoss, self).__init__()
        self.reduction = reduction
        self.data_range = data_range

    def forward(self, img1, img2):
        """
        Computes the Multi-Scale SSIM loss between two images.

        Args:
            img1 (Tensor): Reconstructed images. Shape: (N, C, H, W)
            img2 (Tensor): Target images. Shape: (N, C, H, W)

        Returns:
            Tensor: The computed Multi-Scale SSIM loss.
        """
        ssim_val = ms_ssim_metric(
            img1,
            img2,
            data_range=self.data_range,
            reduction=self.reduction
        )
        return 1 - ssim_val

def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([
        np.exp(-(x - window_size//2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    return gauss

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window @ _1D_window.t()
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_custom(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device)
    padding = window_size // 2  # Ensure padding is integer

    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average

    def forward(self, img1, img2):
        return 1 - ssim_custom(img1, img2, self.window_size, self.size_average)

##################################################
# Decoder Network (Simple Decoder without GAN)
##################################################
class Decoder(nn.Module):
    def __init__(self, encoding_dim, channels=3, feature_map_base=64, image_size=64):
        super(Decoder, self).__init__()
        self.encoding_dim = encoding_dim
        self.channels = channels
        self.image_size = image_size

        # Project and reshape
        self.project = nn.Sequential(
            nn.Linear(encoding_dim, feature_map_base * 8 * 4 * 4),
            nn.ReLU(True),
        )

        # # Transposed convolutional layers to upsample to the desired image size
        # self.main = nn.Sequential(
        #     nn.ConvTranspose2d(feature_map_base * 8, feature_map_base * 4, 4, 2, 1, bias=False),
        #     nn.ReLU(True),
        #
        #     nn.ConvTranspose2d(feature_map_base * 4, feature_map_base * 2, 4, 2, 1, bias=False),
        #     nn.ReLU(True),
        #
        #     nn.ConvTranspose2d(feature_map_base * 2, feature_map_base, 4, 2, 1, bias=False),
        #     nn.ReLU(True),
        #
        #     nn.ConvTranspose2d(feature_map_base, channels, 4, 2, 1, bias=False),
        #     nn.Tanh()  # Output in [-1, 1]
        # )

        self.main = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_map_base * 8, feature_map_base * 4, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_map_base * 4, feature_map_base * 2, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_map_base * 2, feature_map_base, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_map_base, channels, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, enc):
        x = self.project(enc)             # (batch, feature_map_base*8*4*4)
        x = x.view(x.size(0), -1, 4, 4)   # Reshape to (batch, feature_map_base*8, 4, 4)
        out = self.main(x)                # (batch, channels, image_size, image_size)
        return out

##################################################
# Weight Initialization Function
##################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:
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
# Training Function with Reconstruction Loss
##################################################
def trainEncoderDecoder(
        encodings_list,
        images_list,
        encoding_dim=2048,      # Adjusted to match your data
        image_size=64,
        channels=3,
        batch_size=16,
        epochs=20,               # Increased epochs for better training
        lr=1e-5,                # Reduced learning rate
        device="cuda",
        save_interval=5,
        save_decoder_path="decoder.pth"
):
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)

    # Set random seeds for reproducibility
    import random
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)
    set_seed(42)

    # Convert lists to arrays
    encodings_arr = np.stack(encodings_list, axis=0)  # (N, encoding_dim)
    images_arr = np.stack(images_list, axis=0)        # (N, C, H, W)

    # Ensure correct shape
    if images_arr.ndim == 4 and images_arr.shape[-1] in [1, 3]:
        images_arr = images_arr.transpose(0, 3, 1, 2)

    # Create dataset and dataloader
    dataset = EncImgDataset(encodings_arr, images_arr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Instantiate decoder
    decoder = Decoder(
        encoding_dim=encoding_dim,
        channels=channels,
        feature_map_base=64,
        image_size=image_size
    ).to(device)
    decoder.apply(weights_init)

    # Optimizer
    optimizer = optim.Adam(decoder.parameters(), lr=lr, betas=(0.5, 0.999))

    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Loss Function
    criterion = nn.L1Loss()  # You can also use nn.MSELoss()

    from torchvision.models import vgg16
    # Load the VGG16 model and extract its features
    vgg = vgg16(pretrained=True).features[:5].eval()  # Use the first few layers
    vgg = vgg.to(device)  # Move to the appropriate device
    # Freeze the model parameters to prevent updates during training
    for param in vgg.parameters():
        param.requires_grad = False
    def perceptual_loss(output, target):
        # Pass both images through the VGG model
        output_features = vgg(output)
        target_features = vgg(target)
        # Compute the MSE loss between their feature maps
        return nn.functional.mse_loss(output_features, target_features)
    criterion = perceptual_loss

    criterion = SSIMLoss(window_size=11, size_average=True)

    #criterion = MultiScaleSSIMLoss(data_range=1.0, reduction='mean')

    # Initialize scaler for mixed precision
    scaler = amp.GradScaler()

    # Training loop
    for epoch in range(epochs):
        decoder.train()
        epoch_loss = 0.0

        for i, (enc_batch, real_imgs) in enumerate(dataloader):
            enc_batch = enc_batch.to(device)      # (batch, encoding_dim)
            real_imgs = real_imgs.to(device)      # (batch, C, H, W)

            optimizer.zero_grad()

            with amp.autocast():
                # Forward pass
                reconstructed_imgs = decoder(enc_batch)  # (batch, C, H, W)

                # Compute loss
                #loss = criterion(reconstructed_imgs, real_imgs)

                reconstructed_imgs_clamped = torch.clamp((reconstructed_imgs + 1) / 2.0, 0.0, 1.0)
                real_imgs_clamped = torch.clamp((real_imgs + 1) / 2.0, 0.0, 1.0)
                loss = criterion(reconstructed_imgs_clamped, real_imgs_clamped)

            # Backward pass
            scaler.scale(loss).backward()

            # Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # =========================================================
            # Monitoring and Debugging Outputs
            # =========================================================
            if i % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

                # Check for NaNs in loss
                if torch.isnan(loss):
                    print(f"NaN detected in loss at Epoch {epoch + 1}, Batch {i}")
                    return

                # Check activations for NaNs
                if torch.isnan(reconstructed_imgs).any():
                    print(f"NaN detected in reconstructed images at Epoch {epoch + 1}, Batch {i}")
                    return

                # Check weights for NaNs
                for name, param in decoder.named_parameters():
                    if torch.isnan(param).any():
                        print(f"NaN detected in parameter '{name}' at Epoch {epoch + 1}, Batch {i}")
                        return

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{epochs}] Average Loss: {avg_epoch_loss:.4f}")

        # Step the scheduler
        scheduler.step()

        # Save checkpoints
        if (epoch + 1) % save_interval == 0:
            torch.save(decoder.state_dict(), f"decoder_epoch_{epoch + 1}.pth")
            print(f"Saved decoder at epoch {epoch + 1}")

    # Save final decoder
    torch.save(decoder.state_dict(), save_decoder_path)
    print(f"Training complete. Final decoder saved to: {save_decoder_path}")

##################################################
# Data Loading and Preprocessing
##################################################
def load_data(csv_path, image_folder, image_size=64):
    df = pd.read_csv(csv_path)

    print(df.columns)
    print("Number of columns:", len(df.columns))
    # Should be 2049 => 1 for image_name + 2048 for features

    encodings_list = []
    images_list = []

    for idx, row in df.iterrows():
        image_name = row["image_name"]  # e.g. "something.png"

        # Extract 2048 features
        feature_values = row.iloc[1:].values.astype(np.float32)  # shape (2048,)

        # Check for NaNs or Infs in encodings
        if np.isnan(feature_values).any() or np.isinf(feature_values).any():
            print(f"Warning: Encoding contains NaN or Inf at index {idx}. Skipping.")
            continue

        # Load image
        image_path = os.path.join(image_folder, image_name)
        if not os.path.isfile(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue

        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"Warning: Could not load image: {image_path}. Skipping.")
            continue

        # Resize to 64x64 if needed
        img_bgr = cv2.resize(img_bgr, (image_size, image_size))

        # Convert BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Convert to float
        img_rgb = img_rgb.astype(np.float32)
        # Map [0..255] -> [-1..1]
        img_rgb = img_rgb / 127.5 - 1.0

        # Check for NaNs or Infs in images
        if np.isnan(img_rgb).any() or np.isinf(img_rgb).any():
            print(f"Warning: Image contains NaN or Inf: {image_path}. Skipping.")
            continue

        encodings_list.append(feature_values)
        images_list.append(img_rgb)

    print("Loaded encodings:", len(encodings_list))
    print("Loaded images:", len(images_list))

    return encodings_list, images_list

##################################################
# Image Generation Example
##################################################
def generate_image_from_encoding(
        decoder_path,
        encoding,
        encoding_dim=2048,
        image_size=64,
        channels=3,
        device="cuda",
        save_path="generated_image.png"
):
    # Convert encoding to a Torch tensor of shape (1, encoding_dim)
    encoding_t = torch.from_numpy(encoding).float().unsqueeze(0).to(device)

    # Initialize decoder
    decoder = Decoder(
        encoding_dim=encoding_dim,
        channels=channels,
        feature_map_base=64,
        image_size=image_size
    ).to(device)
    decoder.apply(weights_init)

    # Load the saved weights
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.eval()

    # Generate the image
    with torch.no_grad():
        fake_image_t = decoder(encoding_t)  # shape (1, C, 64, 64)

    # Convert from tensor to NumPy for OpenCV or other usage
    # If your decoder uses Tanh, values are in [-1,1]. Map them to [0..255].
    fake_image_t = fake_image_t.squeeze(0).cpu()  # (C, H, W)
    fake_image_np = (fake_image_t.numpy() + 1) / 2.0  # now in [0..1]
    fake_image_np = (fake_image_np * 255.0).clip(0, 255).astype(np.uint8)  # (C, H, W)

    # Transpose to (H, W, C) for OpenCV
    fake_image_np = np.transpose(fake_image_np, (1, 2, 0))  # (64, 64, 3) in RGB

    # Convert RGB -> BGR for OpenCV
    fake_image_bgr = cv2.cvtColor(fake_image_np, cv2.COLOR_RGB2BGR)

    fake_image_bgr = cv2.resize(fake_image_bgr,(64,192))
    # Save the generated image
    cv2.imwrite(save_path, fake_image_bgr)
    print(f"Generated image saved to {save_path}")

##################################################
# Main Execution
##################################################
if __name__ == "__main__":
    # Paths
    csv_path = "D:/bcc/gan_test/bioencodings.csv"
    image_folder = "D:/bcc/gan_test/encodings_images"

    # Load data
    encodings_list, images_list = load_data(csv_path, image_folder, image_size=64)

    # Check if data is loaded
    if len(encodings_list) == 0 or len(images_list) == 0:
        raise ValueError("No data loaded. Please check the CSV file and image paths.")

    #Start Training
    trainEncoderDecoder(
        encodings_list=encodings_list,
        images_list=images_list,
        encoding_dim=2048,   # <--- KEY
        image_size=64,
        channels=3,
        batch_size=16,
        epochs=10,            # Increased epochs for better reconstruction
        lr=1e-5,              # Reduced learning rate for stability
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_interval=5,
        save_decoder_path="decoder_final.pth"
    )

    ##################################################
    # Image Generation Example
    ##################################################
    # Choose an index to generate image from encoding

    # indices = [100,120,304,504,302,401]
    #
    # for index in indices:
    #     if index >= len(encodings_list):
    #         raise IndexError(f"Index {index} is out of bounds for encodings list of length {len(encodings_list)}.")
    #
    #     random_encoding = encodings_list[index]  # shape (2048,)
    #
    #     # Generate image from encoding
    #     generate_image_from_encoding(
    #         decoder_path="decoder_final.pth",
    #         encoding=random_encoding,
    #         encoding_dim=2048,
    #         image_size=64,
    #         channels=3,
    #         device="cuda" if torch.cuda.is_available() else "cpu",
    #         save_path="generated_image.png"
    #     )
    #
    #     generated_image_bgr = cv2.imread("generated_image.png")
    #
    #     # Optionally, display the real image for comparison
    #     real_image = images_list[index]  # (C, H, W)
    #
    #     cv2.imshow("Real Image", real_image)
    #     cv2.imshow("Gen Image", generated_image_bgr)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    import pandas as pd
    import numpy as np
    import os


    def compute_mean_encodings_and_generate_images(
            records_csv_path,
            bioencodings_csv_path,
            decoder_path,
            output_folder,
            encoding_dim=2048,
            image_size=64,
            channels=3,
            device="cuda"
    ):
        # Load records and bioencodings
        records_df = pd.read_csv(records_csv_path)
        bioencodings_df = pd.read_csv(bioencodings_csv_path)

        # Ensure proper naming alignment (add .png if necessary)
        records_df["image_name"] = records_df["img_id"] + ".png"

        # Merge records with bioencodings to align images
        merged_df = records_df.merge(
            bioencodings_df, on="image_name", how="inner"
        )  # Inner join to match encodings only

        if merged_df.empty:
            raise ValueError("No matching images found between records and bioencodings.")

        # Extract species and encoding columns
        encoding_columns = [str(i) for i in range(encoding_dim)]
        grouped = merged_df.groupby("species")[encoding_columns]

        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Compute mean encodings and generate images for each species
        for species, encodings in grouped:

            if len(encodings) < 5:
                print(f"Skipping species '{species}' with less than 5 encodings.")
                continue

            mean_encoding = encodings.mean().values.astype(np.float32)  # Mean encoding

            # Generate image and save it
            save_path = os.path.join(output_folder, f"{species}.png")
            generate_image_from_encoding(
                decoder_path=decoder_path,
                encoding=mean_encoding,
                encoding_dim=encoding_dim,
                image_size=image_size,
                channels=channels,
                device=device,
                save_path=save_path,
            )
            print(f"Generated image for species '{species}' saved to {save_path}.")


    # File paths
    records_csv_path = "D:/bcc/gan_test/records_with_metrics.csv"
    bioencodings_csv_path = "D:/bcc/gan_test/bioencodings.csv"
    decoder_path = "decoder_final.pth"
    output_folder = "D:/bcc/gan_test/generated_species_images"

    # Run the function
    compute_mean_encodings_and_generate_images(
        records_csv_path=records_csv_path,
        bioencodings_csv_path=bioencodings_csv_path,
        decoder_path=decoder_path,
        output_folder=output_folder,
        encoding_dim=2048,
        image_size=64,
        channels=3,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )