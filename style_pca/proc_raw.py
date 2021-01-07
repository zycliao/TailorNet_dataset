"""
1. Convert raw garment displacement to mesh
(the same displacement to different meshes of different genders)
Here, we use chest-flattened SMPL body, so that the garment mesh doesn't have chest bump
2. Smooth all garments and save their mesh and displacement
INPUT: {ROOT}/raw_data/{garment_class}.npy
OUTPUT: {ROOT}/{garment_class}_{gender}/pca/raw.npy
                                            smooth.npy
                                            smooth_disp.npy
"""
import os
import os.path as osp
import pickle
import numpy as np
from tqdm import tqdm
from smpl_torch import SMPLNP
from utils.rotation import get_Apose
from utils.diffusion_smoothing import DiffusionSmoothing as DS
from global_var import ROOT


if __name__ == '__main__':
    garment_class = 'pant'
    raw_dir = osp.join(ROOT, 'raw_data')
    # save_dir = osp.join(ROOT, '{}_{}/pca'.format(garment_class, gender))

    smpl = SMPLNP('neutral')
    apose = get_Apose()
    with open(osp.join(ROOT, 'garment_class_info.pkl'), 'rb') as f:
        class_info = pickle.load(f, encoding='latin-1')
    vert_indices = class_info[garment_class]['vert_indices']

    raw_path = osp.join(raw_dir, '{}.npy'.format(garment_class))
    beta_path = osp.join(raw_dir, '{}_betas.npy'.format(garment_class))
    all_disp = np.load(raw_path)
    betas = np.load(beta_path)

    data_num = len(all_disp)
    vbody, vcloth = smpl(betas, np.tile(apose[None], [data_num, 1]),
                         all_disp, garment_class, batch=True)
    canonical_body, _ = smpl(np.zeros([10]), apose, None, None)
    trans = np.mean((canonical_body[None] - vbody)[:, vert_indices, :], 1, keepdims=True)
    trans_cloth = vcloth + trans

    # if you use your own data, make sure trans_cloth is a garment that aligns canonical_body (a-pose, zero beta)

    np.save(osp.join(raw_dir, '{}_trans.npy'.format(garment_class)), trans_cloth)

    # smoothing
    print("Start smoothing")
    ds = DS(trans_cloth[0], class_info[garment_class]['f'])
    smooth_step = 100
    smooth_vs = []
    for unsmooth_v in tqdm(trans_cloth):
        smooth_v = unsmooth_v.copy()
        for _ in range(smooth_step):
            smooth_v = ds.smooth(smooth_v, smoothness=0.03)
        smooth_vs.append(smooth_v)
    smooth_vs = np.array(smooth_vs, dtype=np.float32)

    np.save(osp.join(raw_dir, '{}_smooth.npy'.format(garment_class)), smooth_vs)
