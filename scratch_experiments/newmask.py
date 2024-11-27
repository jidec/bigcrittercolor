import torch
import cv2
import numpy as np
from torchvision import transforms
from pathlib import Path
import segmentation_models_pytorch as smp

# Function to initialize U-Net++ model
def _initializeUNetPP(encoder_name):

    model = smp.UnetPlusPlus(
        encoder_name=encoder_name,        # choose encoder, e.g., 'resnet34'
        encoder_weights="imagenet",      # use pre-trained weights for encoder
        in_channels=3,                   # input channels (e.g., RGB)
        classes=1                        # output channels (binary mask)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model.load_state_dict(torch.load(model_weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model

def applySegmentationMask(image, model, encoder_name="resnet34", threshold=0.5):
    """
    Apply a trained U-Net++ model to a cv2 image and return the masked image.

    Args:
        image (np.ndarray): Input image (H x W x C) in cv2 format (BGR).
        encoder_name (str): Encoder used in U-Net++ (default: 'resnet34').
        threshold (float): Threshold to convert the predicted mask to binary (default: 0.5).

    Returns:
        np.ndarray: Masked cv2 image.
    """

    # Preprocess the input image
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([352, 352]),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()

    # Resize mask back to the original image size
    mask = cv2.resize((mask > threshold).astype(np.uint8), (image.shape[1], image.shape[0]))

    # Apply the mask to the original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

# Path to image and model weights
image_path = "D:/bcc/beetle_appendage_segmenter/test/image/INATRANDOM-161980567.jpg"
model_weights_path = "D:/bcc/beetle_appendage_segmenter/aux_segmenter_unetpp.pt"

# Load the image
image = cv2.imread(image_path)

# Apply the segmentation mask
masked_image = applySegmentationMask(image, model_weights_path)

# Display the result
cv2.imshow("Masked Image", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()