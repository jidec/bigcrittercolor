import os

# quick helper to convert between the path/location of an image (i.e. "D:/data/masks/INAT-23131_mask.jpg")
#   and its name (i.e. "INAT-23131_mask.jpg"
def _imgPathToName(path):
    other, filename = os.path.split(path)
    return filename