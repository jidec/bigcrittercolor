
def _imgNameToID(name):
    name = name.replace('.jpg','')
    name = name.replace('.png','')
    return name.split("_")[0]
