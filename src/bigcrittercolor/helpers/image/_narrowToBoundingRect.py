import cv2

# narrow a BGR binary image to a bounding rect around the white portion
def _narrowToBoundingRect(img, return_img_and_bb=False):
  if len(img.shape) == 2:
    return img

  img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # create bounding rect around img
  th = cv2.threshold(img_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  coords = cv2.findNonZero(th)
  x, y, w, h = cv2.boundingRect(coords)

  # narrow image to bounding rect
  img = img[y:y + h, x:x + w]

  if return_img_and_bb:
    return(img,(x,y,w,h))
  return img