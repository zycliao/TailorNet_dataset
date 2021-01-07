import os
import os.path as osp
import numpy as np
from tqdm import tqdm

from global_var import ROOT
from utils.rotation import get_Apose, interpolate_pose
from smpl_torch import SMPLNP_Lres
import shutil
import time
from simulation_pose.gen_pose_utils import \
    Dataset, is_intersec, find_index, calc_dis
from utils.part_body import part_body_faces


def     gen_pose(pose_num, betas, nfold=1, div_thresh=0.1, DEBUG=False, gender='neutral',
             intersection=True, garment_class=None):
    """
    if beta is not None, we will discard frames with self-intersection
    """
    t0 = time.time()
    apose = get_Apose()
    smpl = SMPLNP_Lres(gender=gender)
    if garment_class is None:
        faces = smpl.base.faces
    else:
        faces = part_body_faces(garment_class)

    dataset = Dataset()
    data_num = len(dataset)
    poses = np.copy(dataset.poses)

    beta_num = len(betas)
    if intersection:
        all_verts = []
        for beta in betas:
            verts = smpl(np.tile(np.expand_dims(beta, 0), [poses.shape[0], 1]), poses, batch=True)
            all_verts.append(verts)
        good_poses = []
        for pi in tqdm(range(len(poses))):
            p = poses[pi]
            no_intersec = True
            for beta_idx in range(beta_num):
                v = all_verts[beta_idx][pi]
                if is_intersec(v, faces):
                    no_intersec = False
                    break
            if no_intersec:
                good_poses.append(p)
        poses = np.array(good_poses)
        data_num = poses.shape[0]

    if 0 < pose_num < data_num:
        data_num = pose_num*nfold
        random_idx = np.arange(data_num)
        np.random.shuffle(random_idx)
        random_idx = random_idx[:data_num]
        poses = poses[random_idx]

    all_poses = np.copy(poses)
    all_data_num = data_num
    npose_pfold = int(np.ceil(1.*all_data_num / nfold))
    all_thetas = []
    all_pose_orders = []
    for fold_i in range(nfold):
        poses = np.copy(all_poses[fold_i*npose_pfold: (fold_i+1)*npose_pfold])
        data_num = len(poses)
        verts, joints = smpl(np.tile(np.expand_dims(betas[0]*0, 0), [data_num, 1]), poses, batch=True, return_J=True)

        _, apose_joints = smpl(betas[0]*0, apose, return_J=True)

        pose_num = joints.shape[0]
        chosen_mask = np.zeros([joints.shape[0]], dtype=np.bool)
        dist = calc_dis(joints, joints, garment_class)

        a_dist = calc_dis(np.expand_dims(apose_joints, 0), joints, garment_class)[0]
        closest = np.argmin(a_dist)
        chosen_mask[closest] = True
        pose_order = [-2, closest]
        dist_path = [a_dist[closest]]
        thetas = [apose, poses[closest]]

        print(pose_num)
        for c in tqdm(range(pose_num - 1)):
            last_idx = pose_order[-1]
            cur_dist = dist[last_idx]
            cur_dist[np.where(chosen_mask)] = np.inf
            closest = np.argmin(cur_dist)
            d = cur_dist[closest]
            finished = False
            if d > div_thresh:
                if not intersection:
                    div_num = int(d / div_thresh)
                    inter_poses = interpolate_pose(poses[last_idx],
                                                   poses[closest], div_num)
                    inter_poses = inter_poses[1: -1]
                else:
                    cur_dist_copy = np.copy(cur_dist)
                    while True:
                        closest = np.argmin(cur_dist_copy)
                        d = cur_dist_copy[closest]
                        if np.isinf(d):
                            finished = True
                            break
                        cur_dist_copy[closest] = np.inf
                        # interpolate and see if there's self-intersection
                        div_num = int(d / div_thresh)
                        inter_poses = interpolate_pose(poses[last_idx],
                                                       poses[closest], div_num)
                        inter_poses = inter_poses[1: -1]

                        all_verts = []
                        for beta in betas:
                            vs = smpl(np.tile(np.expand_dims(beta, 0), [len(inter_poses), 1]), inter_poses, batch=True)
                            all_verts.append(vs)
                        intersec = False
                        for vs in all_verts:
                            for v in vs:
                                if is_intersec(v, faces):
                                    intersec = True
                                    break
                            if intersec:
                                break
                        # if there's self-intersection, we should find a second closest pose. Otherwise, break the loop
                        if not intersec:
                            break
                        # print(c)
                if not finished:
                    for dd in range(div_num):
                        pose_order.append(-1)
                        dist_path.append(-1)
                        thetas.append(inter_poses[dd])
            if not finished:
                chosen_mask[closest] = True
                pose_order.append(closest)
                dist_path.append(cur_dist[closest])
                thetas.append(poses[closest])
            else:
                break
        pose_order = np.array(pose_order)
        thetas = np.array(thetas)
        dur = time.time() - t0
        print("theta num: {}".format(thetas.shape[0]))
        print("time: {} s".format(dur))

        glob_pose_order = find_index(dataset.poses, thetas)
        all_thetas.append(thetas)
        all_pose_orders.append(glob_pose_order)
    return all_thetas, all_pose_orders


if __name__ == '__main__':
    garment_class = 'skirt'
    genders = ['female']
    TEST = True
    for gender in genders:

        GEN_POSE =  True
        POSE_NUM = 50
        STABILITY_FRAMES = 2
        if TEST:
            batch_num = 1
        else:
            batch_num = 18
        # 0.1 for upper clothes. 0.05 for lower clothes
        DIV_THRESH = 0.1
        lowest = -2

        smpl = SMPLNP_Lres(gender=gender)
        A_theta = get_Apose()

        num_betas = 10 if gender == 'neutral' else 300
        root_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender))

        # read betas and gammas
        betas = np.load(osp.join(root_dir, 'shape', 'betas.npy'))
        gammas = np.load(osp.join(root_dir, 'style', 'gammas.npy'))

        for beta_i, beta in enumerate(betas):

            save_dir = os.path.join(root_dir, 'pose', 'beta_{:03d}'.format(beta_i))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            pose_label_paths = [os.path.join(save_dir, 'poses_{:03d}.npz'.format(k)) for k in range(batch_num)]
            exist = True
            for pose_label_path in pose_label_paths:
                if not os.path.exists(pose_label_path):
                    exist = False
            if TEST:
                assert exist, pose_label_paths
            if not exist:
                all_thetas, all_pose_order = gen_pose(POSE_NUM, beta[None], nfold=batch_num, div_thresh=DIV_THRESH,
                                                      gender=gender, garment_class=garment_class)
                for pose_label_path, thetas, pose_order in zip(pose_label_paths, all_thetas, all_pose_order):
                    np.savez(pose_label_path, pose_order=pose_order, thetas=thetas.astype(np.float32))

            # read pivots
            if TEST:
                pivots_path = osp.join(root_dir, 'test.txt')
            else:
                pivots_path = osp.join(root_dir, 'pivots.txt')
            if osp.exists(pivots_path):
                with open(pivots_path) as f:
                    pivots = f.read().strip().splitlines()
                pivots = [k.split('_') for k in pivots]
            else:
                continue

            for beta_str, gamma_str in pivots:
                if int(beta_str) != beta_i:
                    continue
                pivots_pose_dir = os.path.join(root_dir, 'pose', '{}_{}'.format(beta_str, gamma_str))
                if not os.path.exists(pivots_pose_dir):
                    os.makedirs(pivots_pose_dir)
                for bi, pose_label_path in enumerate(pose_label_paths):
                    pivots_pose_path = os.path.join(pivots_pose_dir, 'poses_{:03d}.npz'.format(bi))
                    if not os.path.exists(pivots_pose_path):
                        shutil.copy(pose_label_path, pivots_pose_path)

