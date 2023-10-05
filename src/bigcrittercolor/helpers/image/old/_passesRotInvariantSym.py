import cv2
import numpy as np
from bigcrittercolor.helpers.imgtransforms import _getRotInvariantBlobSymmetry

def _passesRotInvariantSym(rgb_mask, sym_min, show=False):
    greyu8 = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    sym = _getRotInvariantBlobSymmetry(greyu8,show=show)
    if sym < sym_min:
        return False
    else:
        return True

