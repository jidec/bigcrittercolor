import cv2
import numpy as np

def _imgIsValid(img):
    # Check if either image is None
    if img is None:
        print("Error: One or both images are None.")
        return False

    # Check if either image has dimensions of 0x0xX
    if img.size == 0:
        print("Error: One or both images have no size.")
        return False

    # Check if image is all black
    if np.all(img == 0):
        print("Error: One or both images are all black.")
        return False

    return True

#img = np.random.randint(0, 255, (100, 110, 3))
#black_img = np.zeros((100, 110, 3))
#none_img = None
#mask = np.random.randint(0, 2, (100,110), dtype=np.uint8)
#none_mask = None
#mask_wrongsize = np.random.randint(0, 2, (100,100), dtype=np.uint8)
#mask_32 = np.random.randint(0, 2, (100,100), dtype=np.uint32)

#print(_imgIsValid(img))
#print(_imgIsValid(black_img))
#print(_imgIsValid(mask))
#print(_imgIsValid(none_img))
#print(_imgIsValid(none_mask))
#print(_imgAndMaskAreValid(img,mask_32))