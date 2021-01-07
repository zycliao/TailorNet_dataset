# remove the feet to prevent self interpenetration
import os
import numpy as np
from global_var import ROOT
from utils.ios import read_obj


def part_body_faces(gc):
    path = os.path.join(ROOT, 'smpl', 'part_{}.obj'.format(gc))
    _, f = read_obj(path)
    return np.array(f, dtype=np.int32)