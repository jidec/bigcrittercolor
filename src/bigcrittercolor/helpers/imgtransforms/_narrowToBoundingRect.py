import cv2

# narrow a binary image to a bounding rect around the white portion
def _narrowToBoundingRect(img):
  img_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  # create bounding rect around img
  th = cv2.threshold(img_binary, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
  coords = cv2.findNonZero(th)
  x, y, w, h = cv2.boundingRect(coords)

  # narrow image to bounding rect
  img = img[y:y + h, x:x + w]
  return (img)