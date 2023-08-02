from PIL import Image
import numpy as np

# resize a list of images to their average dimensional size
def _resizeImgsToAvgSize(image_list):
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