"""
Go through the scanning dataset and find available registrations for each garment class
save their displacement (in one .npy file) and meta information (triangulation, mapping with SMPL)
INPUT: registration mesh under /BS/bharat-2/work/data/renderings
OUTPUT: {ROOT}/raw_data/{garment_class}.npy
        {ROOT}/garment_class_info.pkl
"""
import os
import os.path as osp
import json
import numpy as np
import pickle
from utils.ios import read_obj
from smpl_torch import SMPLNP
from tqdm import tqdm
from utils.rotation import get_Apose
from utils.diffusion_smoothing import DiffusionSmoothing as DS
from global_var import *

APOSE = get_Apose()

v_num = {'t-shirt': 7702, 'shirt': 9723, 'pant': 4718, 'short-pant': 2710, 'coat': 10116, 'skirt': 7130}


def get_verts(RAW_DIR):
    verts = []
    people_names = [os.path.splitext(k)[0] for k in os.listdir(RAW_DIR)]
    people_names = list(set(people_names))
    faces = None
    for people_name in people_names:
        obj_path = osp.join(RAW_DIR, '{}.obj'.format(people_name))

        if not os.path.exists(obj_path):
            continue
        v, faces_ = read_obj(obj_path)
        if faces is None:
            faces = faces_
        verts.append(v)

    return people_names, verts, faces


if __name__ == '__main__':
    garment_class = 'skirt'
    RAW_DIR = osp.join(ROOT, 'skirt_reg')
    SAVE_DIR = osp.join(ROOT, 'raw_data')

    people_names, verts, faces = get_verts(RAW_DIR)
    n_verts = verts[0].shape[0]

    smpl = SMPLNP('female')
    apose = get_Apose()
    canonical_body, _ = smpl(np.zeros([300]), apose, None, None)
    with open(osp.join(ROOT, 'garment_class_info.pkl'), 'rb') as f:
        class_info = pickle.load(f, encoding='latin-1')
    pant_ind = class_info['pant']['vert_indices']
    pant_bnd = np.load(osp.join(ROOT, 'pant_upper_boundary.npy'))
    skirt_bnd = np.load(osp.join(ROOT, 'skirt_upper_boundary.npy'))
    pant_bnd_loc = np.mean(canonical_body[pant_ind][pant_bnd], 0)

    all_v = []
    for people_name, v in zip(people_names, verts):
        skirt_bnd_loc = np.mean(v[skirt_bnd], 0)
        trans = (pant_bnd_loc - skirt_bnd_loc)[None]
        trans_v = v + trans
        all_v.append(trans_v)
    all_v = np.array(all_v).astype(np.float32)
    np.save(os.path.join(SAVE_DIR, '{}.npy'.format(garment_class)), all_v)
    np.save(os.path.join(SAVE_DIR, '{}_trans.npy'.format(garment_class)), all_v)

    # smoothing
    ds = DS(all_v[0], faces)
    smooth_step = 100
    smooth_vs = []
    for unsmooth_v in tqdm(all_v):
        smooth_v = unsmooth_v.copy()
        for _ in range(smooth_step):
            smooth_v = ds.smooth(smooth_v, smoothness=0.03)
        smooth_vs.append(smooth_v)
    smooth_vs = np.array(smooth_vs, dtype=np.float32)

    np.save(osp.join(SAVE_DIR, '{}_smooth.npy'.format(garment_class)), smooth_vs)

    # save garment meta information (triangulation and vert_indices for each garment class)
    garment_meta_path = os.path.join(SAVE_DIR, '..', 'garment_class_info.pkl')
    if os.path.exists(garment_meta_path):
        with open(garment_meta_path, 'rb') as f:
            garment_meta = pickle.load(f, encoding='latin-1')
    else:
        garment_meta = {}
    if garment_class not in garment_meta:
        garment_meta[garment_class] = {'vert_indices': [-1]*n_verts, 'f': faces}
    with open(garment_meta_path, 'wb') as f:
        pickle.dump(garment_meta, f)
