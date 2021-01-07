import os
import os.path as osp
import numpy as np

from global_var import ROOT
from utils.rotation import get_Apose, interpolate_pose
from smpl_torch import SMPLNP_Lres
from utils.ios import save_pc2, read_pc2
from utils.part_body import part_body_faces


if __name__ == '__main__':
    garment_class = 'skirt'
    gender = 'female'

    TEST = True
    STABILITY_FRAMES = 2
    if TEST:
        batch_num = 1
    else:
        batch_num = 18
    # 0.1 for upper clothes. 0.05 for lower clothes
    lowest = -2

    smpl = SMPLNP_Lres(gender=gender)
    apose = get_Apose()
    num_betas = 10 if gender == 'neutral' else 300
    vcanonical = smpl(np.zeros([num_betas]), apose)

    root_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender))
    ss_dir = osp.join(root_dir, 'style_shape')

    # read pivots
    if TEST:
        pivots_path = osp.join(root_dir, 'test.txt')
    else:
        pivots_path = osp.join(root_dir, 'pivots.txt')
    with open(pivots_path) as f:
        pivots = f.read().strip().splitlines()
    pivots = [k.split('_') for k in pivots]

    # read betas and gammas
    betas = np.load(osp.join(root_dir, 'shape', 'betas.npy'))
    gammas = np.load(osp.join(root_dir, 'style', 'gammas.npy'))


    # pivots = [['000', '000'], ['001', '000']]
    for beta_str, gamma_str in pivots:
        beta_gamma = "{}_{}".format(beta_str, gamma_str)
        save_dir = os.path.join(root_dir, 'pose', beta_gamma)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        beta = betas[int(beta_str)]
        gamma = gammas[int(gamma_str)]
        vbeta = smpl(beta, apose)

        res_dir = os.path.join(save_dir, 'res')
        if not os.path.isdir(res_dir):
            os.makedirs(res_dir)
            os.chmod(res_dir, 0o774)

        # shape transition, read from style_shape
        if garment_class in ['pant', 'skirt', 'short-pant']:
            transition_path = osp.join(ss_dir, f'motion_beta{beta_str}_gamma{gamma_str}.pc2')
        else:
            transition_path = osp.join(ss_dir, f'motion_{beta_str}.pc2')
        transition_verts = read_pc2(transition_path)

        for bi in range(batch_num):
            pose_label_path = os.path.join(save_dir, 'poses_{:03d}.npz'.format(bi))
            if not osp.exists(pose_label_path):
                print("{} doesn't exist. Skip it".format(pose_label_path))
                continue
            dst_path = os.path.join(save_dir, 'motion_{:03d}.pc2'.format(bi))
            if osp.exists(dst_path):
                print("{} already exists. Skip it".format(dst_path))
                continue
            pose_label = np.load(pose_label_path)

            # poses
            thetas = pose_label['thetas']
            verts = smpl(np.tile(np.expand_dims(beta, 0), [thetas.shape[0], 1]), thetas, batch=True)
            verts = np.tile(np.expand_dims(verts, 1), [1, STABILITY_FRAMES + 1, 1, 1])
            verts = np.reshape(verts, [-1, 6890, 3])
            verts[:, :, 1] -= lowest

            verts = np.concatenate([transition_verts, verts], 0)
            save_pc2(verts, dst_path)
