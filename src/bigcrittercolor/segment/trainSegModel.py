from pathlib import Path
import click
import torch
from sklearn.metrics import f1_score, roc_auc_score
import copy
import csv
import time
import numpy as np
import torch
from tqdm import tqdm
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from typing import Any, Callable, Optional
from PIL import Image
from torchvision.datasets.vision import VisionDataset
#from torchvision.transforms import AutoAugmentPolicy
import os

def trainSegModel(training_dir_location, num_epochs=8, batch_size=6, num_workers=0, data_transforms=None, img_color_mode="rgb",
                      criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8)), resnet_type="deeplabv3", resnet_size="101", pretrained=True,
                      print_steps=True, print_details=False):
    """
        Load and train a segmentation model

        :param str training_dir_name: the name of the training dir within data/other/training_dirs
        :param int num_epochs: the number of epochs to train for
        :param int batch_size: the number of images to train on per batch
        :param int num_workers: the number of workers
        :param str model_name: what to call the outputted model.pt
        :param Dictionary data_transforms: instructions for transforming the data, see default example
    """
    if print_steps: print("Started loadTrainSegModel")

    if data_transforms is None:
        data_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(224), # crop a random part of the image and resize it to size 224
            transforms.Resize([344, 344]),
            # transforms.Resize([448, 448]),
            #transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            # transforms.AutoAugment(),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #transforms.Grayscale()
        ])
        if print_steps: print("Loaded default transforms")

    # download the deeplabv3 resnet101 model which is pretrained on a subset
    # of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset
    # TODO- allow specifying a model
    if resnet_size == "50":
        if resnet_type == "deeplabv3":
            model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained,
                                                            progress=True)
        elif resnet_type == "maskrcnn":
            model = models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    elif resnet_size == "101":
        model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained,
                                                       progress=True)
    # set the model head
    # change output channel size to greater than 1 for multichannel segmentation
    model.classifier = DeepLabHead(2048, 1)

    # put the model in training mode
    model.train()

    # create the experiment directory if not present
    # exp_directory = proj_dir + "/data/ml_models/logs"
    # not exp_directory.exists():
    #    exp_directory.mkdir()

    # specify the loss function
    # criterion = torch.nn.MSELoss(reduction='mean')

    # specify the optimizer with a learning rate
    # TODO - allow specifying a learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # specify the evaluation metrics
    metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}

    # create the dataloader
    # dataloaders = getDataloaders(
    #     data_directory, batch_size=batch_size)
    # _ = trainModel(model,
    #                 criterion,
    #                 dataloaders,
    #                 optimizer,
    #                 bpath=exp_directory,
    #                 metrics=metrics,
    #                 num_epochs=num_epochs)

    image_datasets = {
        x: _SegmentationDataset(root=Path(training_dir_location) / x,
                               transforms=data_transforms,
                               image_folder="Image",
                               mask_folder="Mask",
                               image_color_mode=img_color_mode)
        for x in ['Train', 'Test']
    }
    if print_steps: print("Created datasets object")

    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)
        for x in ['Train', 'Test']
    }
    if print_steps: print("Created dataloaders object")

    if print_steps: print("Training model...")
    _trainModel(model,
               criterion,
               dataloaders,
               optimizer,
               bpath=training_dir_location,
               metrics=metrics,
               num_epochs=num_epochs,
               name='segmenter')

    # save the trained model
    dest = training_dir_location + '/segmenter.pt'
    torch.save(model, dest)
    if print_steps: print("Saved model to " + dest)

class _SegmentationDataset(VisionDataset):
    """A PyTorch dataset for image segmentation task.
    The dataset is compatible with torchvision transforms.
    The transforms passed would be applied to both the Images and Masks.
    """
    def __init__(self,
                 root: str,
                 image_folder: str,
                 mask_folder: str,
                 transforms: Optional[Callable] = None,
                 seed: int = None,
                 fraction: float = None,
                 subset: str = None,
                 image_color_mode: str = "rgb",
                 mask_color_mode: str = "grayscale") -> None:
        """
        Args:
            root (str): Root directory path.
            image_folder (str): Name of the folder that contains the images in the root directory.
            mask_folder (str): Name of the folder that contains the masks in the root directory.
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            seed (int, optional): Specify a seed for the train and test split for reproducible results. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to None.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
            image_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'rgb'.
            mask_color_mode (str, optional): 'rgb' or 'grayscale'. Defaults to 'grayscale'.

        Raises:
            OSError: If image folder doesn't exist in root.
            OSError: If mask folder doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
            ValueError: If image_color_mode and mask_color_mode are either 'rgb' or 'grayscale'
        """
        super().__init__(root, transforms)
        image_folder_path = Path(self.root) / image_folder
        mask_folder_path = Path(self.root) / mask_folder
        if not image_folder_path.exists():
            raise OSError(f"{image_folder_path} does not exist.")
        if not mask_folder_path.exists():
            raise OSError(f"{mask_folder_path} does not exist.")

        if image_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{image_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )
        if mask_color_mode not in ["rgb", "grayscale"]:
            raise ValueError(
                f"{mask_color_mode} is an invalid choice. Please enter from rgb grayscale."
            )

        self.image_color_mode = image_color_mode
        self.mask_color_mode = mask_color_mode

        if not fraction:
            self.image_names = sorted(image_folder_path.glob("*"))
            self.mask_names = sorted(mask_folder_path.glob("*"))
        else:
            if subset not in ["Train", "Test"]:
                raise (ValueError(
                    f"{subset} is not a valid input. Acceptable values are Train and Test."
                ))
            self.fraction = fraction
            self.image_list = np.array(sorted(image_folder_path.glob("*")))
            self.mask_list = np.array(sorted(mask_folder_path.glob("*")))
            if seed:
                np.random.seed(seed)
                indices = np.arange(len(self.image_list))
                np.random.shuffle(indices)
                self.image_list = self.image_list[indices]
                self.mask_list = self.mask_list[indices]
            if subset == "Train":
                self.image_names = self.image_list[:int(
                    np.ceil(len(self.image_list) * (1 - self.fraction)))]
                self.mask_names = self.mask_list[:int(
                    np.ceil(len(self.mask_list) * (1 - self.fraction)))]
            else:
                self.image_names = self.image_list[
                    int(np.ceil(len(self.image_list) * (1 - self.fraction))):]
                self.mask_names = self.mask_list[
                    int(np.ceil(len(self.mask_list) * (1 - self.fraction))):]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> Any:
        image_path = self.image_names[index]
        mask_path = self.mask_names[index]
        with open(image_path, "rb") as image_file, open(mask_path,
                                                        "rb") as mask_file:
            image = Image.open(image_file)
            if self.image_color_mode == "rgb":
                image = image.convert("RGB")
            elif self.image_color_mode == "grayscale":
                image = image.convert("L")
                image = image.convert("RGB")
            mask = Image.open(mask_file)
            if self.mask_color_mode == "rgb":
                mask = mask.convert("RGB")
            elif self.mask_color_mode == "grayscale":
                mask = mask.convert("L")
            sample = {"image": image, "mask": mask}
            if self.transforms:
                sample["image"] = self.transforms(sample["image"])
                sample["mask"] = self.transforms(sample["mask"])
            return sample

def _trainModel(model, criterion, dataloaders, optimizer, metrics, bpath,
                num_epochs, name):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs['out'], masks)
                    y_pred = outputs['out'].data.cpu().numpy().ravel()
                    y_true = masks.data.cpu().numpy().ravel()
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            # Use a classification threshold of 0.1
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true > 0, y_pred > 0.1))
                        else:
                            batchsummary[f'{phase}_{name}'].append(
                                metric(y_true.astype('uint8'), y_pred))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, name + '_log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
