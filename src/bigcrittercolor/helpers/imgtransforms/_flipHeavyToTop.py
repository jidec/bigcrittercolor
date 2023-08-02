import cv2

# given an image flip the side with more nonblack pixels to the top
def _flipHeavyToTop(img):
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