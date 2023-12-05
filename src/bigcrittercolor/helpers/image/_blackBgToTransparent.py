import cv2

# converts a 3-channel BGR or RGB image to a 4-channel PNG by making completely black pixels transparent
def _blackBgToTransparent(img):
    # Make sure the image has an alpha channel
    if img.shape[2] < 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Set the alpha channel to fully transparent for black pixels
    img[(img[:, :, 0] == 0) & (img[:, :, 1] == 0) & (img[:, :, 2] == 0), 3] = 0

    return img
