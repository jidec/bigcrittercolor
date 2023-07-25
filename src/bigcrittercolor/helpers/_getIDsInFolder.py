import os
from bigcrittercolor.helpers import _imgNameToID

def _getIDsInFolder(folder):
    files = os.listdir(folder)
    ids = [_imgNameToID(f) for f in files]
    return(ids)