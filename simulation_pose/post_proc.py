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
from utils.ios import read_pc2
from global_var import ROOT


if __name__ == '__main__':
    garment_class = 'skirt'
    gender = 'female'
    lowest = -2
    STABILITY_FRAMES = 2

    smpl = SMPLNP(gender)
    apose = torch.from_numpy(get_Apose().astype(np.float32))
    data_root = osp.join(ROOT, '{}_{}'.format(garment_class, gender))
    pose_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender), 'pose')
    ss_dir = osp.join(data_root, 'style_shape')
    shape_dir = osp.join(data_root, 'shape')

    beta_strs = [k.replace('.obj', '') for k in os.listdir(shape_dir) if k.endswith('.obj')]
    betas = np.load(osp.join(data_root, 'shape', 'betas.npy'))

    all_ss = [k for k in os.listdir(pose_dir) if len(k) == 7]
    for ss in all_ss:
        beta_str, gamma_str = ss.split('_')
        pose_ss_dir = osp.join(pose_dir, ss)

        if garment_class in ['pant', 'skirt', 'short-pant']:
            transition_path = osp.join(ss_dir, f'motion_beta{beta_str}_gamma{gamma_str}.pc2')
        else:
            transition_path = osp.join(ss_dir, f'motion_{beta_str}.pc2')
        transition_verts = read_pc2(transition_path)
        transition_num = transition_verts.shape[0]

        result_names = [k for k in os.listdir(pose_ss_dir) if k.startswith('result_') and k.endswith('.pc2')]
        for result_name in result_names:
            batch_str = result_name.replace('result_', '').replace('.pc2', '')
            garment_path = osp.join(pose_ss_dir, result_name)
            save_path = osp.join(pose_ss_dir, 'unposed_{}.npy'.format(batch_str))
            if os.path.exists(save_path):
                continue
            theta_path = osp.join(pose_ss_dir, 'poses_{}.npz'.format(batch_str))
            thetas = np.load(theta_path)['thetas'].astype(np.float32)
            garment_vs = read_pc2(garment_path)
            if garment_vs is None:
                print("{} is broken".format(result_name))
                continue
            garment_vs[:, :, 1] += lowest
            beta = betas[int(beta_str)]

            all_unposed = []
            for theta_i, theta in enumerate(thetas):
                ii = len(transition_verts)+theta_i*(STABILITY_FRAMES+1)+STABILITY_FRAMES
                v = garment_vs[ii]
                unposed_v = smpl.base.forward_unpose_deformation(torch.from_numpy(theta).unsqueeze(0), torch.from_numpy(beta).unsqueeze(0),
                                                                 torch.from_numpy(v.astype(np.float32)).unsqueeze(0), garment_class)
                unposed_v = unposed_v[0].detach().cpu().numpy()
                all_unposed.append(unposed_v)
            all_unposed = np.array(all_unposed, dtype=np.float32)
            np.save(save_path, all_unposed)
            print(save_path)
