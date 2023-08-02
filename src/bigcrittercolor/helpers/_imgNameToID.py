
# quick helper to convert between the name of an image (i.e. "INAT-23131_mask.jpg") and its ID (i.e. "INAT-23131")
def _imgNameToID(name):
    name = name.replace('.jpg','')
    name = name.replace('.png','')
    return name.split("_")[0]
