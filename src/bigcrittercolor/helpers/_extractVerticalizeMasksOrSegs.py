import cv2
import numpy as np
import math
from PIL import Image

# extract a list of masks (binary) or segments (color) from their backgrounds
#   THEN rotate them to be vertical by fitting an ellipse, drawing a line through it,
#    and transforming the image to be vertical using the line
#    We ALSO normalize the size of the images
# verticalize - to rotate a mask or segment to be pointing upward with the "heaviest" side of the mask or segment at the top

# if trim_side_method==True this function also wraps _vertTrimSides, a replacement for the standard approach
#   that verticalizes using a different method (based on the longest stretch of white pixels in the mask) then trims the sides
#   of the bounding box with the intention of getting rid of wings and other appendages
def _extractVerticalizeMasksOrSegs(imgs, size_normalize=True, trim_sides_method=False, show=False):
  for index, img in enumerate(imgs):
    start_img = np.copy(img)

    # if image is greyscale convert it to RGB
    if len(img.shape) == 2:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # trim sides on axis is an alternative to normal verticalize method
    if trim_sides_method:
      img = _vertTrimSides(img)

    # otherwise do normal method by ellipse fitting
    else:
      # threshold the mask or segment
      ret, thresh = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)

      if thresh is not None:
        # make sure it's greyscale
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

        # find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # get the biggest contour and fit an ellipse to it
        big_contour = max(contours, key = cv2.contourArea)

        big_ellipse = cv2.fitEllipse(big_contour)

        # get params from ellipse
        (xc,yc),(d1,d2),angle = big_ellipse

        # compute major radius
        rmajor = max(d1,d2)/2
        if angle > 90:
            angle = angle - 90
        else:
            angle = angle + 90
        xtop = xc + math.cos(math.radians(angle))*rmajor
        ytop = yc + math.sin(math.radians(angle))*rmajor
        xbot = xc + math.cos(math.radians(angle+180))*rmajor
        ybot = yc + math.sin(math.radians(angle+180))*rmajor
        line = [(xtop,ytop),(xbot,ybot)]

        if(math.isnan(xtop) or math.isnan(ytop) or math.isnan(xbot) or math.isnan(ybot)):
          return(None)

        img = _vert_img_using_line(img, line)

        img = _narrow_to_bounding_rect(img)

        img = _flip_heavy_to_top(img)

        imgs[index] = img

  if size_normalize:
    imgs = _resize_images_to_average_size(imgs)

  return imgs

def _vertTrimSides(binary_img, trim_distance=30, show=False):
  line = _getLongestLineThruBlob(binary_img, show=show)

  # vert the image
  binary_img = _vert_img_using_line(binary_img, line)

  if binary_img is None:
    return None
  if show:
    cv2.imshow("Vert Img", binary_img)
    cv2.waitKey(0)

  # vert the image WITH a line
  img_with_line = np.copy(binary_img)
  cv2.line(img_with_line, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (255), 5)

  # create empty image and draw line
  line_img = np.copy(binary_img)
  cv2.line(line_img, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), (255), 5)

  # narrow image to bounding rect
  img = _narrow_to_bounding_rect(binary_img)
  # narrow image WITH line to bounding rect
  img_with_line = _narrow_to_bounding_rect(img)

  # find the topmost white pixel of the vert'ed image with line
  # find the coordinates of white pixels
  white_pixel_coords = cv2.findNonZero(img_with_line)

  if white_pixel_coords is not None:
    # Sort the white pixels by y-coordinate (top to bottom)
    sorted_white_pixels = sorted(white_pixel_coords, key=lambda coord: coord[0][1])

    # Get the x-coordinate of the topmost white pixel
    topmost_white_pixel_x = sorted_white_pixels[0][0][0]

  # get bottom and top halves of image
  bot_half = img[int(img.shape[0] / 2):img.shape[0]]
  top_half = img[0:int(img.shape[0] / 2)]

  # get number of pixels in each
  n_px_bot = bot_half[bot_half > 0].size
  n_px_top = top_half[top_half > 0].size

  # if bottom is heavier than top, flip it good
  if n_px_bot > n_px_top:
    img = cv2.flip(img, 0)

  # trim edges
  x1, y1 = topmost_white_pixel_x - trim_distance, 0
  x2, y2 = topmost_white_pixel_x + trim_distance, np.shape(img)[0]

  # crop the image using NumPy slicing
  cropped_image = img[y1:y2, x1:x2]

  if cropped_image.size == 0:
    return None

  if show:
    cv2.imshow("Sides Trimmed", cropped_image)
    cv2.waitKey(0)

  return cropped_image


# finds the longest line that can be drawn through the sole white blob in a binary image by:
# 1. getting the minimum area rectangle
# 2. finding all points in the top face, AND the bot face
# 3. drawing lines between every top face point and every bottom face point
# 4. finding the line containing the most white pixels (the longest line)

# returns a tuple of points for the line

def _getLongestLineThruBlob(binary_img, line_point_interval=10, show=False):
  # find contours in the binary image
  contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  if len(contours) == 0:
    print("No contour found, skipping getting longest line...")
  else:
    # find the largest contour (the white blob)
    largest_contour = max(contours, key=cv2.contourArea)

    # fit a minimum area rectangle around the contour
    min_area_rect = cv2.minAreaRect(largest_contour)

    # draw the minimum area rectangle on the original image (optional)
    box = cv2.boxPoints(min_area_rect).astype(int)

    # display the image with the minimum area rectangle (optional)
    if show:
      img_with_box = np.copy(binary_img)
      cv2.drawContours(img_with_box, [box], 0, 255, 2)
      cv2.imshow('Image with Minimum Area Rectangle', img_with_box)
      cv2.waitKey(0)

    rect_points = cv2.boxPoints(min_area_rect)

    # Sort the points by y-coordinate in ascending order
    rect_points = sorted(rect_points, key=lambda x: x[1])

    # The top face of the rectangle is the one with the smallest y-coordinate
    top_face_points = rect_points[:2]  # Assuming the rectangle has four corners
    bot_face_points = rect_points[2:4]  # Assuming the rectangle has four corners

    # Draw the top face on a copy of the original image
    result_image = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result_image, [np.int0(top_face_points)], 0, (0, 255, 0), 2)

    # Display the result
    if show:
      cv2.imshow('Top Face of Rectangle', result_image)
      cv2.waitKey(0)

    top_line_coordinates = _points_on_line(top_face_points[0], top_face_points[1])
    top_line_coordinates = top_line_coordinates[0::line_point_interval]

    bot_line_coordinates = _points_on_line(bot_face_points[0], bot_face_points[1])
    bot_line_coordinates = bot_line_coordinates[0::line_point_interval]

    tops = []
    bots = []
    nwhite = []

    for t in top_line_coordinates:
      for b in bot_line_coordinates:
        n = _count_white_pixels_along_line(binary_img, start_coords=tuple(t), end_coords=tuple(b))
        tops.append(t)
        bots.append(b)
        nwhite.append(n)

    maxindex = nwhite.index(max(nwhite))
    maxtop = tops[maxindex]
    maxbot = bots[maxindex]

    if show:
      line_img = np.copy(binary_img)
      cv2.line(line_img, maxtop, maxbot, (0, 255, 255), 2)
      cv2.imshow("Image with found line", line_img)
      cv2.waitKey(0)

    return ((maxtop, maxbot))


# get the points on a line
def _points_on_line(start, end):
  idx = np.round(np.array(start)).astype(int)
  end_idx = np.round(np.array(end)).astype(int)
  points = [idx]

  if np.all(idx == end_idx):
    return points

  diff = np.array(end, dtype=float) - np.array(start, dtype=float)
  direction = (diff / np.abs(diff)).astype(int)
  coord = np.array(start, dtype=float)

  while np.any(idx != end_idx):
    # compute how far we need to go to reach the side of the pixel at idx
    t = (idx + direction / 2 - coord) / diff
    i = np.argmin(t)
    coord += t[i] * diff
    idx = idx.copy()
    idx[i] += direction[i]
    points.append(idx)

  return points


# count the number of white pixels in a line from the start to end
def _count_white_pixels_along_line(image, start_coords, end_coords):
  # create a copy of the input image to avoid modifying the original image
  image_copy = np.copy(image)

  # define line coordinates
  x1, y1 = start_coords
  x2, y2 = end_coords

  # Bresenham's line algorithm to get the coordinates of the line pixels
  dx = abs(x2 - x1)
  dy = abs(y2 - y1)
  sx = 1 if x1 < x2 else -1
  sy = 1 if y1 < y2 else -1
  err = dx - dy

  white_pixel_count = 0

  while True:
    # check if the current coordinates are within the image boundaries
    if x1 >= 0 and y1 >= 0 and x1 < image.shape[1] and y1 < image.shape[0]:
      # get the pixel value at the current coordinates
      pixel_value = image_copy[y1, x1]

      # assuming the image is in grayscale, check if the pixel is white (255)
      if pixel_value == 255:
        white_pixel_count += 1

    if x1 == x2 and y1 == y2:
      break

    e2 = 2 * err
    if e2 > -dy:
      err -= dy
      x1 += sx
    if e2 < dx:
      err += dx
      y1 += sy

  return white_pixel_count


def _resize_images_to_average_size(image_list):
  # Compute the average height and width of the images
  total_height = 0
  total_width = 0
  num_images = len(image_list)

  for image in image_list:
    height, width = image.shape[:2]
    total_height += height
    total_width += width

  avg_height = total_height // num_images
  avg_width = total_width // num_images

  # Resize each image to the average size
  resized_images = []
  for image in image_list:
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize((avg_width, avg_height), Image.ANTIALIAS)
    resized_images.append(np.array(resized_image))

  return resized_images


# warp an image to vertical using a line to be made vertical
# line must be of the format [(xtop,ytop),(xbot,ybot)]
def _vert_img_using_line(img, line, show=False):
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


def _narrow_to_bounding_rect(img):
  img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # create bounding rect around img
  th = cv2.threshold(img_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  coords = cv2.findNonZero(th)
  x, y, w, h = cv2.boundingRect(coords)

  # narrow image to bounding rect
  img = img[y:y + h, x:x + w]
  return (img)


def _flip_heavy_to_top(img):
  # get bottom and top halves of image
  bot_half = img[int(img.shape[0] / 2):img.shape[0]]
  top_half = img[0:int(img.shape[0] / 2)]

  # get number of pixels in each
  n_px_bot = bot_half[bot_half > 0].size
  n_px_top = top_half[top_half > 0].size

  # if bottom is heavier than top, flip it good
  if n_px_bot > n_px_top:
    img = cv2.flip(img, 0)
  return img