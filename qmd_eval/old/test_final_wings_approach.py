from bigcrittercolor.segment import inferMasks
from bigcrittercolor.segment import trainSegModel
import torch
from torchvision import transforms

# 101 size deeplab, batch 6, rgb
#trainSegModel(training_dir_location="D:/bcc/dfly_aux_segmenter", img_color_mode="rgb",
#             num_epochs=10, batch_size=6, num_workers=0, resnet_type="deeplabv3", resnet_size="101",
#              pretrained=True, print_steps=True, print_details=False)

# 101 size deeplab, batch 6, greyscale
#trainSegModel(training_dir_location="D:/bcc/dfly_aux_segmenter2", img_color_mode="grayscale",
#              num_epochs=10, batch_size=6, num_workers=0, resnet_type="deeplabv3", resnet_size="101",
#              pretrained=True, print_steps=True, print_details=False)

# 50 size deeplab, batch 6, rgb
#trainSegModel(training_dir_location="D:/bcc/dfly_aux_segmenter3", img_color_mode="rgb",
#              num_epochs=10, batch_size=6, num_workers=0, resnet_type="deeplabv3", resnet_size="50",
#              pretrained=True, print_steps=True, print_details=False)

# 101 size deeplab, batch 6, rgb non pretrained
# last to finish was 3
#trainSegModel(training_dir_location="D:/bcc/dfly_aux_segmenter4", img_color_mode="rgb",
#              num_epochs=10, batch_size=6, num_workers=0, resnet_type="deeplabv3", resnet_size="101",
#              pretrained=False, print_steps=True, print_details=False)

# 101 size deeplab, batch 6, rgb, loss 6
#trainSegModel(training_dir_location="D:/bcc/dfly_aux_segmenter5", img_color_mode="rgb",
#              num_epochs=10, batch_size=6, num_workers=0, resnet_type="deeplabv3", resnet_size="101",
#              pretrained=False, print_steps=True, print_details=False, criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6)))


# whichever ends up best plus AutoAugment
#transforms = transforms.Compose([
#            transforms.Resize([344, 344]),
#            transforms.AutoAugment(),
#            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
#        ])
#trainSegModel(training_dir_location="D:/bcc/dfly_aux_segmenter6", img_color_mode="rgb", transforms=transforms,
#              num_epochs=10, batch_size=6, num_workers=0, resnet_type="deeplabv3", resnet_size="101",
#              pretrained=False, print_steps=True, print_details=False, criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6)))

#inferMasks(strategy="prompt1", text_prompt="insect", data_folder="D:/bcc/dfly_appr_expr/appr_final",
#aux_segmodel_location="D:/bcc/dfly_aux_segmenter/segmenter.pt")

transforms = transforms.Compose([
            transforms.Resize([344, 344]),
            # transforms.Resize([448, 448]),
            transforms.RandomHorizontalFlip(), # horizontally flip with 50% prob
            transforms.RandomRotation(degrees=(0,360)),
            #transforms.RandomAdjustSharpness(sharpness_factor=0.2),
            #transforms.RandomAutocontrast(),
            transforms.ToTensor(),  # convert a pil image or numpy ndarray to tensor
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #transforms.Grayscale()
        ])
trainSegModel(training_dir_location="D:/bcc/dfly_aux_segmenter6", img_color_mode="rgb", data_transforms=transforms,
              num_epochs=12, batch_size=11, num_workers=0, resnet_type="deeplabv3", resnet_size="101",
              pretrained=False, print_steps=True, print_details=False, criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6)))