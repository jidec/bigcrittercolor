
# quick helper to convert between the name of an image (i.e. "INAT-23131_mask.jpg") and its ID (i.e. "INAT-23131")
def _imgIDToObsID(img_id):
    # Split the name at each dash
    parts = img_id.split('-')
    # Join all parts except the last one
    obs_id = '-'.join(parts[:-1])

    return obs_id
