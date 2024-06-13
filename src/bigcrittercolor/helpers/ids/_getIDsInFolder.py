import os
import random

from bigcrittercolor.helpers.ids import _imgNameToID

def _getIDsInFolder(folder,sample_n=None):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    ids = [_imgNameToID(f) for f in files]
    if sample_n is not None:
        ids = random.sample(ids, sample_n)
    return(ids)