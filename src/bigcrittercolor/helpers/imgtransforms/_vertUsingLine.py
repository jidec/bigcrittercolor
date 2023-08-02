import cv2
import numpy as np

# verticalize an image using a line
# the provided line is rotated to the vertical and the image is transformed to the same vertical
# the line basically represents the axis on which the image should rest at the end
def _vertUsingLine(img, line, show=False):
  if show:
    cv2.imshow("Image to Vert", img)
    cv2.waitKey(0)

  # create empty image and draw line
  line_img = np.zeros_like(img, dtype=np.uint8)

  cv2.line(line_img, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (255, 255, 255), 1)

  if show:
    cv2.imshow("Line Image", line_img)
    cv2.waitKey(0)

  line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

  edges = cv2.Canny(line_img, 50, 150, apertureSize=3)

  # get line in HoughLines format
  lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # Lower the threshold

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
    img = cv2.warpAffine(img, M, (nW, nH))

  return img