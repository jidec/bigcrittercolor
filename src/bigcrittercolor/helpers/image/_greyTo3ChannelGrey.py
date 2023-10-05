import cv2

def _greyTo3ChannelGrey(img):
    img = cv2.merge([img, img, img])
    return img