"""
translate and unpose simulation results
"""
import os
import os.path as osp
import torch
import numpy as np
import trimesh
from smpl_torch import SMPLNP
from utils.rotation import get_Apose
from utils.ios import write_obj
from global_var import ROOT

if __name__ == '__main__':
    garment_class = 'skirt'
    gender = 'female'
    lowest = -2

    import pickle
    with open(os.path.join(ROOT, 'garment_class_info.pkl'), 'rb') as f:
        class_info = pickle.load(f, encoding='latin-1')
    garment_f = class_info[garment_class]['f']

    smpl = SMPLNP(gender)
    apose = torch.from_numpy(get_Apose().astype(np.float32))
    data_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender), 'style_shape')
    shape_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender), 'shape')

    betas = np.load(osp.join(shape_dir, 'betas.npy'))


    with open(osp.join(data_dir, '../avail.txt')) as f:
        all_ss = f.read().strip().splitlines()
    for ss in all_ss:
        beta_str, gamma_str = ss.split('_')
        style_shape = f'beta{beta_str}_gamma{gamma_str}'
        style_shape_path = osp.join(data_dir, style_shape+'.obj')
        style_shape_save_path = osp.join(data_dir, style_shape+'.npy')
        beta = betas[int(beta_str)]

        m = trimesh.load(style_shape_path, process=False)
        v = m.vertices
        v[:, 1] += lowest

        unposed_v = smpl.base.forward_unpose_deformation(apose.unsqueeze(0), torch.from_numpy(beta).unsqueeze(0),
                                                         torch.from_numpy(v.astype(np.float32)).unsqueeze(0), garment_class)
        unposed_v = unposed_v[0].detach().cpu().numpy()
        np.save(style_shape_save_path, unposed_v)
