import numpy as np
import cv2
from bigcrittercolor.helpers.image import _vertUsingLine, _getLongestLineThruBlob2, _getBlobEllipseLine, _narrowToBoundingRect, _flipHeavyToTop, _getRotInvariantBlobSymmetry
from bigcrittercolor.helpers.imgtransforms._getBlobEllipseLine import _getBlobEllipseLine
from bigcrittercolor.helpers.imgtransforms._cropImgSides import _cropImgSides
from bigcrittercolor.helpers.imgtransforms._getRotInvariantBlobSymmetry import _getRotInvariantBlobSymmetry
from bigcrittercolor.helpers import _showImages

# verticalize an image either using an ellipse or using the longest blob axis
# COLOR IMAGE is taken
# happens to wrap trim sides on axis functionality, may want to separate that out
def _verticalizeImg(img, strategy="polygon", polygon_e_mult=0.01, bound=True, flip=True, sym_if_sym=None, crop_sides_size = None, input_line=None, return_line=False, show=False):

  start_img = np.copy(img)

  greyu8 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.uint8) # convert to greyscale uint8

  if sym_if_sym is not None:
    sym_score = _getRotInvariantBlobSymmetry(greyu8,show=show)
    print(sym_score)
    if sym_score > sym_if_sym:
      strategy = "symmetry"
  match strategy:
    case "polygon":
      line = _getLongestLineThruBlob2(greyu8, show=show, epsilon_mult=polygon_e_mult)
    case "ellipse":
      line = _getBlobEllipseLine(greyu8)
    case "symmetry":
      img = _getRotInvariantBlobSymmetry(greyu8, return_sym_img=True, show=show)

  if return_line:
    return line
  if input_line is not None:
    line = input_line

  if strategy != "symmetry":
    vert_img = _vertUsingLine(img, line, show)
    img = np.copy(vert_img)

  if bound:
    bounded = _narrowToBoundingRect(img)
    img = np.copy(bounded)

  if flip:
    flipped = _flipHeavyToTop(img)
    img = np.copy(flipped)
    _showImages(show,[start_img, vert_img, bounded, flipped],titles=["Start","Verticalized","Bounded","Flipped"])

  if crop_sides_size is not None:
    white_pixel_indices = np.where(img != 0)
    highest_white_pixel_y = np.min(white_pixel_indices[0])
    highest_white_pixel_x = np.min(white_pixel_indices[1][np.where(white_pixel_indices[0] == highest_white_pixel_y)])
    img = _cropImgSides(img, middle_x=highest_white_pixel_x,lr_crop_px=crop_sides_size)

  return img