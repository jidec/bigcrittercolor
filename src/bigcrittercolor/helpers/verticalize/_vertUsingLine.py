import cv2
import numpy as np
from bigcrittercolor.helpers import _showImages

# verticalize an image using a line
# the provided line is rotated to the vertical and the image is transformed to the same vertical
# the line basically represents the axis on which the image should rest at the end
# input image must be RGB, BGR, or 3-channel greyscale
def _vertUsingLine(img, line, return_img_and_line=False, show=False):

  _showImages._showImages(show,img,"Image to Vert")

  # create empty image and draw line
  line_img = np.zeros_like(img, dtype=np.uint8)

  # avoid error caused by NaNs in line
  if np.isnan(line[0][0]) or np.isnan(line[0][1]) or np.isnan(line[1][0] or np.isnan(line[1][1])):
    return img

  cv2.line(line_img, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (255,255,255), 3)
  line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

  #edges = cv2.Canny(line_img, 50, 150, apertureSize=3)

  # get line in HoughLines format
  lines = cv2.HoughLines(line_img, 1, np.pi / 180, 50)  # Lower the threshold
  hough_img = np.zeros_like(img,dtype=np.uint8)

  # visualize HoughLines
  if lines is not None:
    for line in lines:
      rho, theta = line[0]
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a * rho
      y0 = b * rho
      x1 = int(x0 + 1000 * (-b))
      y1 = int(y0 + 1000 * (a))
      x2 = int(x0 - 1000 * (-b))
      y2 = int(y0 - 1000 * (a))
      cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

  #_showImages(show, [img, line_img,edges,hough_img], ["Image", "Line Image", "Edges Image","Hough Image"])

  if lines is not None:
    # get new line parameters
    rho, theta = lines[0][0]
    angle = theta * 180 / np.pi

    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # get rotation matrix and transform
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # warp image to new rotation
    vert_img = cv2.warpAffine(img, M, (nW, nH))
    _showImages._showImages(show, [img, line_img, hough_img,vert_img], ["Image", "Line Image", "Hough Image","Vert Image"])

    if return_img_and_line:
      line_img = cv2.warpAffine(line_img, M, (nW, nH))
      y_coords, x_coords = np.where(line_img == 255)
      max_index = np.argmax(y_coords)
      min_index = np.argmin(y_coords)

      start_coord = (x_coords[min_index], y_coords[min_index])
      end_coord = (x_coords[max_index], y_coords[max_index])
      return (vert_img, (start_coord,end_coord))
    else:
      return vert_img

  else:
    return img