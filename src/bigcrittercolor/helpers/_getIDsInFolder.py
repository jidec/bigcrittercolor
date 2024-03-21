import os
import random

from bigcrittercolor.helpers import _imgNameToID

def _getIDsInFolder(folder,sample_n=None):
    files = os.listdir(folder)
    ids = [_imgNameToID(f) for f in files]
    if sample_n is not None:
        ids = random.sample(ids, sample_n)
    return(ids)