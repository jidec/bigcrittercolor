import os

def pathToImgID(path):
    other, filename = os.path.split(path)
    return filename