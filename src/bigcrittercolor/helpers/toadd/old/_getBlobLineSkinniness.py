import numpy as np
import cv2
from bigcrittercolor.helpers import _showImages

# get the skinniness of an axis of a blob using a line drawn through that axis 
def _getBlobLineSkinniness(binary_img, line_start, line_end, show=False):

    line_img = np.zeros_like(binary_img)
    num_blob_pix = np.sum(binary_img == 255)
    #print(num_blob_pix)

    # Find contours
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Get the contour with the largest area (assuming it's the main blob)
    c = max(contours, key=cv2.contourArea)
    # Get the minimum area rectangle
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # Extract the ROI from the another image using bounding rectangle for simplicity
    x, y, w, h = cv2.boundingRect(c)
    line_img = line_img[y:y + h, x:x + w]
    binary_img = binary_img[y:y + h, x:x + w]

    cv2.line(line_img,line_start,line_end,color=255,thickness=int(num_blob_pix / 1000))

    masked = cv2.bitwise_and(binary_img, binary_img, mask=line_img.astype(np.uint8)) #mask=cv2.cvtColor(line_img,cv2.COLOR_RGB2GRAY).astype(np.uint8))
    num_white_pix_line = np.sum(line_img == 255)
    num_white_pix_masked = np.sum(masked == 255)

    # starts at 1 and probably maxes at 1.5
    skinny = num_white_pix_line / num_white_pix_masked


    _showImages(show,[binary_img,line_img,masked],["Image","Line", str(skinny) + " Masked"])

    return(skinny)