import numpy as np
import cv2
from bigcrittercolor.helpers.imgtransforms import _vertUsingLine, _getLongestLineThruBlob, _getBlobEllipseLine, _narrowToBoundingRect, _flipHeavyToTop
from bigcrittercolor.helpers.imgtransforms._getBlobEllipseLine import _getBlobEllipseLine
from bigcrittercolor.helpers.imgtransforms._chopImgLR import _chopImgLR

# verticalize an image either using an ellipse or using the longest blob axis
# COLOR IMAGE is taken
def _verticalizeImg(img, by_ellipse=False, crop_lr_size = None, show=False):
  start_img = np.copy(img)

  greyu8 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.uint8)
  # if image is greyscale convert it to RGB
  #if len(img.shape) == 2:
  #  img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

  if by_ellipse:
    line = _getBlobEllipseLine(greyu8)
  else:
    line = _getLongestLineThruBlob(greyu8,show=show)

  img = _vertUsingLine(img,line)

  img = _narrowToBoundingRect(img)

  img = _flipHeavyToTop(img)

  if crop_lr_size is not None:
    white_pixel_indices = np.where(img != 0)
    highest_white_pixel_y = np.min(white_pixel_indices[0])
    highest_white_pixel_x = np.min(white_pixel_indices[1][np.where(white_pixel_indices[0] == highest_white_pixel_y)])
    img = _chopImgLR(img, middle_x=highest_white_pixel_x,n_chop_px=crop_lr_size)

  return img