import copy
import csv
import time
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Any, Callable, Optional
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
import segmentation_models_pytorch as smp  # Import for U-Net

from sklearn.metrics import f1_score, roc_auc_score

from bigcrittercolor.helpers import _SegmentationDataset

# Define a dictionary of available segmentation models
MODEL_DICT = {
    'deeplabv3_resnet50': smp.DeepLabV3,
    'deeplabv3_resnet101': smp.DeepLabV3,
    'unet': smp.Unet,  # Add U-Net model
    # You can add more architectures as needed
}

def trainSegModel(training_dir_location, model_name="unetpp", encoder_name="resnet34", num_epochs=8, batch_size=6,
                  num_workers=0, data_transforms=None, img_color_mode="rgb", criterion=None,
                  print_steps=True, print_details=False):
    """
    Load and train a segmentation model with various architectures and detailed performance feedback.
    """
    if print_steps:
        print("Starting trainSegModel...")

    if data_transforms is None:
        data_transforms = transforms.Compose([
            transforms.Resize([352, 352]),
            transforms.ToTensor(),
        ])
        if print_steps:
            print("Loaded default transforms")

    # Load the specified model architecture without pretraining
    #model = load_model(model_name, encoder_name)
    if model_name == "unetpp":
        model = smp.UnetPlusPlus(encoder_name=encoder_name, encoder_weights="imagenet", in_channels=3, classes=1)
    elif model_name == "unet":
        model = smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet", in_channels=3, classes=1)
    else:
        raise ValueError("Model name not supported. Choose 'unet' or 'unetpp'.")

    # Specify the loss function if not provided
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    image_datasets = {
        x: _SegmentationDataset._SegmentationDataset(root=Path(training_dir_location) / x,
                                transforms=data_transforms,
                                image_folder="Image",
                                mask_folder="Mask",
                                image_color_mode=img_color_mode)
        for x in ['Train', 'Test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)
        for x in ['Train', 'Test']
    }

    if print_steps:
        print("Training model...")
    model = _trainModel(model, criterion, dataloaders, optimizer, metrics, training_dir_location,
                        num_epochs=num_epochs, name='segmenter')

    # Save the trained model
    dest = Path(training_dir_location) / 'segmenter.pt'
    torch.save(model.state_dict(), dest)
    if print_steps:
        print(f"Saved model to {dest}")

def load_model(model_name, encoder_name):
    """Load the specified model from MODEL_DICT without pretraining."""
    if model_name not in MODEL_DICT:
        raise ValueError(f"Model '{model_name}' not supported. Available options: {list(MODEL_DICT.keys())}")
    if model_name == 'unet':
        # Load U-Net without pretraining
        model = MODEL_DICT[model_name](encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
    else:
        # For other models, also set encoder_weights to None to avoid pretraining
        model = MODEL_DICT[model_name](encoder_name=encoder_name, encoder_weights=None)
    return model

# Continue with your _trainModel, _SegmentationDataset, and other helper functions...



def _trainModel(model, criterion, dataloaders, optimizer, metrics, bpath, num_epochs, name):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    history = {'epoch': [], 'train_loss': [], 'test_loss': [], 'train_f1': [], 'test_f1': [], 'train_auroc': [],
               'test_auroc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        epoch_losses, epoch_f1s, epoch_aurocs = {}, {}, {}  # Temporary dicts to store values

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()
            else:
                model.eval()

            running_loss, y_true, y_pred = 0.0, [], []
            for sample in tqdm(dataloaders[phase]):
                inputs, masks = sample['image'].to(device), sample['mask'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, masks)
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                y_true.append(masks.cpu().numpy().ravel())
                y_pred.append(outputs.detach().cpu().numpy().ravel())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)

            # Calculate metrics
            f1 = f1_score(y_true > 0, y_pred > 0.1)
            auroc = roc_auc_score(y_true.astype('uint8'), y_pred)

            # Store metrics for the current phase
            epoch_losses[phase] = epoch_loss
            epoch_f1s[phase] = f1
            epoch_aurocs[phase] = auroc

            print(f"{phase} Loss: {epoch_loss:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
            if phase == 'Test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            # Show random sample of masks for test phase
            if phase == 'Test':
                _plotRandomTestSamples(model=model,dataloader=dataloaders[phase], device=device)

        # After both phases, store to history once per epoch
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_losses['Train'])
        history['test_loss'].append(epoch_losses['Test'])
        history['train_f1'].append(epoch_f1s['Train'])
        history['test_f1'].append(epoch_f1s['Test'])
        history['train_auroc'].append(epoch_aurocs['Train'])
        history['test_auroc'].append(epoch_aurocs['Test'])

    model.load_state_dict(best_model_wts)
    plot_metrics(history, bpath)
    return model


def plot_metrics(history, bpath):
    """Plot and save training metrics over epochs."""
    epochs = history['epoch']
    plt.figure(figsize=(12, 6))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')

    # F1 and AUROC plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_f1'], label='Train F1 Score')
    plt.plot(epochs, history['test_f1'], label='Test F1 Score')
    plt.plot(epochs, history['train_auroc'], label='Train AUROC')
    plt.plot(epochs, history['test_auroc'], label='Test AUROC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('F1 Score and AUROC')

    plt.tight_layout()
    save_path = Path(bpath) / "metrics.png"
    plt.savefig(save_path)
    print(f"Saved training metrics to {save_path}")


import torch
from torchvision import transforms
import numpy as np
from PIL import Image


def get_segmentation_mask(model, image: Image.Image, device: torch.device) -> np.ndarray:
    """
    Given a preloaded segmentation model and an image, returns the predicted mask.

    Parameters:
        model (torch.nn.Module): The loaded segmentation model in evaluation mode.
        image (PIL.Image.Image): The input image.
        device (torch.device): The device to run the model on (CUDA or CPU).

    Returns:
        np.ndarray: The binary mask (or probability map) predicted by the model.
    """
    # Define the required transformations
    preprocess = transforms.Compose([
        transforms.Resize((352, 352)),  # Resize to match model's expected input
        transforms.ToTensor(),
    ])

    # Preprocess the image
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    input_tensor = input_tensor.to(device)

    # Run the model on the input image
    with torch.no_grad():  # No gradient needed for inference
        output = model(input_tensor)

    # Move output to CPU and remove batch dimension
    mask = output.squeeze(0).cpu().numpy()

    # Optional: Convert to binary mask with threshold (for example, threshold=0.5)
    mask = (mask > 0.5).astype(np.uint8)  # Binary mask with thresholding

    return mask

import random


def _plotRandomTestSamples(model, dataloader, device, threshold=0.5, num_samples=3):
    """Show random sample of test images with predicted masks using the specified threshold."""
    model.eval()
    with torch.no_grad():
        # Get a random batch of samples from the dataloader
        sample_batch = next(iter(dataloader))
        images, true_masks = sample_batch['image'].to(device), sample_batch['mask'].to(device)

        # Make predictions on the sample batch
        outputs = model(images)

        # Select random indices to display
        indices = random.sample(range(images.size(0)), min(num_samples, images.size(0)))

        # Plot images, ground truth masks, and predicted masks
        plt.figure(figsize=(15, 5 * num_samples))
        for i, idx in enumerate(indices):
            image = images[idx].cpu().permute(1, 2, 0).numpy()  # Convert to HWC for plotting
            true_mask = true_masks[idx].cpu().squeeze().numpy()
            pred_mask = (outputs[idx].cpu().squeeze().numpy() > threshold).astype(np.uint8)

            # Display the input image
            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.imshow(image)
            plt.title("Input Image")
            plt.axis("off")

            # Display the ground truth mask
            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.imshow(true_mask, cmap="gray")
            plt.title("Ground Truth Mask")
            plt.axis("off")

            # Display the predicted mask with threshold
            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(pred_mask, cmap="gray")
            plt.title(f"Predicted Mask (Threshold: {threshold})")
            plt.axis("off")

        plt.tight_layout()
        plt.show()


import numpy as np
from sklearn.metrics import f1_score, jaccard_score  # F1 Score, IoU
import torch

import numpy as np
from sklearn.metrics import f1_score, jaccard_score  # F1 Score, IoU
import torch


def find_optimal_threshold(model, dataloader, device, num_thresholds=50):
    """
    Find the optimal threshold for binary segmentation.

    Parameters:
        model (torch.nn.Module): Trained segmentation model.
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation set.
        device (torch.device): Device to run the model on.
        num_thresholds (int): Number of thresholds to test (e.g., 50).

    Returns:
        float: Optimal threshold value.
    """
    model.eval()
    thresholds = np.linspace(0, 1, num_thresholds)
    f1_scores = []
    iou_scores = []

    with torch.no_grad():
        for threshold in thresholds:
            all_f1, all_iou = [], []
            for sample in dataloader:
                images, true_masks = sample['image'].to(device), sample['mask'].to(device)
                outputs = model(images)
                preds = outputs.squeeze(1).cpu().numpy()  # Remove batch and channel dimensions

                # Flatten the predictions and true masks for evaluation
                for i in range(len(preds)):
                    # Binarize predictions and true masks based on the current threshold
                    pred_mask = (preds[i] > threshold).astype(np.uint8).ravel()
                    true_mask = (true_masks[i].cpu().numpy() > 0.5).astype(np.uint8).ravel()

                    # Calculate F1 and IoU scores for each thresholded mask
                    all_f1.append(f1_score(true_mask, pred_mask))
                    all_iou.append(jaccard_score(true_mask, pred_mask))

            # Store the average scores for the current threshold
            f1_scores.append(np.mean(all_f1))
            iou_scores.append(np.mean(all_iou))

    # Find the threshold with the highest F1 or IoU score
    best_threshold_f1 = thresholds[np.argmax(f1_scores)]
    best_threshold_iou = thresholds[np.argmax(iou_scores)]

    print(f"Best threshold for F1 Score: {best_threshold_f1}, F1 Score: {max(f1_scores):.4f}")
    print(f"Best threshold for IoU: {best_threshold_iou}, IoU: {max(iou_scores):.4f}")

    _plotRandomTestSamples(model, dataloader, device, best_threshold_f1, 9)

    # Choose the metric you want to optimize, typically IoU or F1 Score
    return best_threshold_f1  # or best_threshold_iou


import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import segmentation_models_pytorch as smp
from bigcrittercolor.helpers import _SegmentationDataset

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model initialization (using U-Net as an example with ResNet34 as encoder)
model_name = 'unet'
encoder_name = 'resnet34'
model = smp.Unet(encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Validation data setup
validation_dir = Path("D:/bcc/beetle_appendage_segmenter/test")  # Replace with your validation data path

# Define transformations for validation
data_transforms = transforms.Compose([
    transforms.Resize((352, 352)),  # Adjust according to your model's input size
    transforms.ToTensor(),
])

# Validation dataset and dataloader
validation_dataset = _SegmentationDataset._SegmentationDataset(
    root=validation_dir,
    transforms=data_transforms,
    image_folder="image",  # Replace with your actual image folder name
    mask_folder="mask",    # Replace with your actual mask folder name
    image_color_mode="rgb"
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=6,   # Adjust as needed
    shuffle=False,
    num_workers=0   # Adjust based on your machine’s capability
)

# Find optimal threshold using the provided function
#threshold = find_optimal_threshold(model, validation_dataloader, device)
#print(f"Optimal threshold: {threshold}")
#threshold = find_optimal_threshold(model, validation_dataloader, device)

# Load your model
#aux_model = torch.load("D:/bcc/beetle_appendage_segmenter/segmenter.pt")
#aux_model.eval()
#aux_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load an image (example path)
#image = Image.open("path/to/your/image.jpg")

# Get the segmentation mask
#mask = get_segmentation_mask(aux_model, image, aux_device)
trainSegModel(training_dir_location="D:/bcc/beetle_appendage_segmenter",num_epochs=30)
