import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, jaccard_score
from PIL import Image
from bigcrittercolor.helpers import _SegmentationDataset
from tqdm import tqdm

from bigcrittercolor.helpers import _bprint
#from bigcrittercolor.segment import _initializeUNetPP

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# function to train U-Net++ model
def trainAuxSegmentationModel(training_dir_location, encoder_name="resnet34", num_epochs=30, batch_size=6,
                  num_workers=0, data_transforms=None, img_color_mode="rgb", criterion=None,
                  print_steps=True, show=True):

    """ Train and save a U-Net++ segmentation model (pretrained on ImageNet).

        This architecture works well as an "auxiliary model" to refine groundedSAM segments, such as to remove wings or get the thorax only.

        The training directory must contain "test" and "train" folders, with each of these having "image" and "mask" folders containing images and correponding masks.

        The images can have any name (without underscores), and the masks must be binary PNG images and have the same name plus "_mask".

            Args:
                training_dir_location (str): location of the training directory
                encoder_name (str): encoder to use with UNet - either "resnet34" or "resnet50"
                num_epochs (int): number of epochs to train for
                batch_size (int): size of the training batch
                num_workers (int): number of CPU cores to use when loading batches
                data_transforms (dict): image augmentations in torchvision transforms format - defaults to minimum transforms.Compose([transforms.Resize([352, 352]), transforms.ToTensor()])
                img_color_mode (rgb): image color mode to use during training
                criterion: a torch loss function - defaults to torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8))
        """

    _bprint(print_steps, "Starting trainAuxSegmentationModel...")

    # default transformations
    if data_transforms is None:
        data_transforms = transforms.Compose([transforms.Resize([352, 352]), transforms.ToTensor()])

    _bprint(print_steps, "Initializing U-Net++ model...")
    # initialize U-Net++ model
    model = _initializeUNetPP(encoder_name)

    # loss function and optimizer
    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # data loaders
    image_datasets = {
        x: _SegmentationDataset._SegmentationDataset(root=Path(training_dir_location) / x, transforms=data_transforms,
                                image_folder="Image", mask_folder="Mask", image_color_mode=img_color_mode)
        for x in ['Train', 'Test']
    }
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers)
                   for x in ['Train', 'Test']}

    _bprint(print_steps, "Training...")
    model = _train(model, criterion, dataloaders, optimizer, num_epochs=num_epochs, print_steps=print_steps,show=show)

    # Save the trained model
    dest = Path(training_dir_location) / 'aux_segmenter_unetpp.pt'
    torch.save(model.state_dict(), dest)
    _bprint(print_steps, f"Saved model to {dest}")

def _train(model, criterion, dataloaders, optimizer, num_epochs, print_steps=True,show=True):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    history = {'epoch': [], 'train_loss': [], 'test_loss': [], 'train_f1': [], 'test_f1': [], 'train_auroc': [],
               'test_auroc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        for phase in ['Train', 'Test']:
            model.train() if phase == 'Train' else model.eval()
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
            f1 = f1_score(y_true > 0, y_pred > 0.5)
            auroc = roc_auc_score(y_true.astype('uint8'), y_pred)

            history[f'{phase.lower()}_loss'].append(epoch_loss)
            history[f'{phase.lower()}_f1'].append(f1)
            history[f'{phase.lower()}_auroc'].append(auroc)

            print(f"{phase} Loss: {epoch_loss:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
            if phase == 'Test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if show:
                    _plotRandomTestSamples(model, dataloaders['Test'], device, epoch=epoch+1)

    if show:
        _plotTrainingMetrics(history=history)
    model.load_state_dict(best_model_wts)
    return model


def _plotRandomTestSamples(model, dataloader, device, epoch, threshold=0.5, num_samples=3):
    """Show random sample of test images with predicted masks using the specified threshold."""
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        images, true_masks = sample_batch['image'].to(device), sample_batch['mask'].to(device)
        outputs = model(images)

        indices = np.random.choice(range(images.size(0)), num_samples, replace=False)
        plt.figure(figsize=(15, 5 * num_samples))
        plt.suptitle(f"Epoch {epoch}", fontsize=16)
        for i, idx in enumerate(indices):
            image = images[idx].cpu().permute(1, 2, 0).numpy()
            true_mask = true_masks[idx].cpu().squeeze().numpy()
            pred_mask = (outputs[idx].cpu().squeeze().numpy() > threshold).astype(np.uint8)

            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.imshow(image)
            plt.title("Input Image")
            plt.axis("off")

            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.imshow(true_mask, cmap="gray")
            plt.title("Ground Truth Mask")
            plt.axis("off")

            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(pred_mask, cmap="gray")
            plt.title(f"Predicted Mask (Threshold: {threshold})")
            plt.axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def _plotTrainingMetrics(history):
    """Plot and save training metrics over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['test_loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')

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
    #save_path = save_dir / "metrics.png"
    #plt.savefig(save_path)
    plt.show()
    #print(f"Saved training metrics to {save_path}")

# Function to initialize U-Net++ model
def _initializeUNetPP(encoder_name='resnet34'):

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,        # choose encoder, e.g., 'resnet34'
        encoder_weights="imagenet",      # use pre-trained weights for encoder
        in_channels=3,                   # input channels (e.g., RGB)
        classes=1                        # output channels (binary mask)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    #model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model

# Example of training the model
#trainAuxSegmentationModel(training_dir_location="D:/bcc/dragonfly_segmenter", num_epochs=15)