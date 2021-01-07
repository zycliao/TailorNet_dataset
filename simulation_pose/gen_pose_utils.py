import os
import numpy as np
import cv2
import time
import pickle

from utils.smpl import SMPLNP
from global_var import ROOT
from utils.rotation import normalize_y_rotation, flip_theta
from utils.rotation import get_Apose, interpolate_pose
# from utils.renderer import Renderer
# from utils.geometry import get_selfintersections
try:
    import pymesh
    from pymesh.selfintersection import detect_self_intersection
    geodis = np.load(os.path.join(ROOT, 'smpl', 'smpl_geodesic.npy'))
except:
    print("Warning: ?")


PAIR_THRESH1 = 160
PAIR_THRESH2 = 140

GEO_THRESH = 0.25

EPS = 1e-6

def find_index(orig_poses, target_poses):
    orig_poses_ext = np.expand_dims(orig_poses, 1)
    target_poses_ext = np.expand_dims(target_poses, 0)
    # (1782, 3003)
    diff = np.sum(np.abs(orig_poses_ext - target_poses_ext), axis=2)
    pose_order = np.argmin(diff, axis=0)
    pose_order[np.where(np.min(diff, axis=0) > EPS)] = -1
    pose_order[0] = -2
    return pose_order


def get_selfintersections(v, f):
    mspy = pymesh.form_mesh(v, f)
    face_pairs = detect_self_intersection(mspy)
    mspy.add_attribute('face_area')
    face_areas = mspy.get_attribute('face_area')
    intersecting_area = face_areas[np.unique(face_pairs.ravel())].sum()
    return face_pairs, intersecting_area


def is_intersec(v, f, thresh=GEO_THRESH):
    pair, _ = get_selfintersections(v, f)
    max_dis = 0
    for p in pair:
        i1 = f[p[0], 0]
        i2 = f[p[1], 0]
        # print(i1)
        # print(i2)
        g = geodis[i1, i2]
        max_dis = np.maximum(g, max_dis)
    return max_dis > thresh

class Dataset(object):
    def __init__(self, split=None):
        """
        pose number: 1782
        if beta is not None, we will discard frames with self-intersection
        """
        pose_npy_path = os.path.join(ROOT, 'pose/pose_norm.npy')
        raw_pose_path = os.path.join(ROOT, 'pose/pose_raw.npy')
        if os.path.exists(pose_npy_path):
            self.poses = np.load(pose_npy_path)
        else:
            pose_dir = os.path.join(ROOT, 'pose', 'SMPL')
            with open(os.path.join(pose_dir, 'male.pkl')) as f:
                mpose = pickle.load(f)
            raw_poses = np.array(mpose)

            flip_poses = flip_theta(raw_poses, batch=True)
            raw_poses = np.stack([raw_poses, flip_poses], 0).reshape((-1, 72))
            raw_poses_copy = np.copy(raw_poses)
            for i, raw_pose in enumerate(raw_poses_copy):
                raw_pose[:3] = normalize_y_rotation(raw_pose[:3])
            norm_poses = raw_poses_copy
            np.save(raw_pose_path, raw_poses)
            np.save(pose_npy_path, norm_poses)
            self.poses = norm_poses
        self.poses_nosplit = np.array(self.poses)
        if split is not None:
            assert split in ['train', 'test']
            split_idx = np.load(os.path.join(ROOT, 'pose/split.npz'))
            self.poses = self.poses[split_idx[split]]

    def get_item(self, idx):
        return self.poses[idx]

    def __len__(self):
        return self.poses.shape[0]

    def debug(self):
        smpl = SMPLNP()
        renderer = Renderer(img_size=512)
        for p in self.poses:
            v, _, _ = smpl(np.zeros(10), p)
            img = renderer(v, smpl.base.faces)
            cv2.imshow('img', img)
            cv2.waitKey()


def calc_dis(j1, j2, garment_class=None):
    valid_joints = {
                    # "smooth_TShirtNoCoat": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
                    # "smooth_ShirtNoCoat": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
                    # "smooth_Pants": [0, 1, 2, 3, 4, 5],
        "t-shirt": [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19],
        "shirt": [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23],
        "pant": [0, 1, 2, 3, 4, 5, 7, 8, 10, 11]
                    }
    j1s = np.expand_dims(j1, 1)  # (bs, 1, jnum, 3)
    j2s = np.expand_dims(j2, 0)  # (1, bs, jnum, 3)
    if garment_class is not None:
        valid_j = valid_joints[garment_class]
        j1s = j1s[:, :, valid_j]
        j2s = j2s[:, :, valid_j]
    dist = np.sqrt(np.sum(np.square(j1s - j2s), -1))
    dist = np.max(dist, -1)
    # (j1_bs, j2_bs)
    return dist


def gen_pose(pose_num, nfold=1, div_thresh=0.1, DEBUG=False, beta=None, gender='neutral',
             intersection=True, garment_class=None):
    """
    if beta is not None, we will discard frames with self-intersection
    """
    t0 = time.time()
    apose = get_Apose()
    smpl = SMPLNP(gender=gender)
    if beta is None:
        beta = np.zeros([10])

    dataset = Dataset()
    data_num = len(dataset)
    poses = np.copy(dataset.poses)

    if intersection:
        verts, _, _ = smpl(np.tile(np.expand_dims(beta, 0), [poses.shape[0], 1]), poses, batch=True)
        good_poses = []
        for v, p in zip(verts, poses):
            if not is_intersec(v, smpl.base.faces):
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
        verts, joints, _ = smpl(np.tile(np.expand_dims(beta, 0), [data_num, 1]), poses, batch=True)

        _, apose_joints, _ = smpl(beta, apose)

        pose_num = joints.shape[0]
        chosen_mask = np.zeros([joints.shape[0]], dtype=np.bool)
        dist = calc_dis(joints, joints, garment_class)

        a_dist = calc_dis(np.expand_dims(apose_joints, 0), joints, garment_class)[0]
        closest = np.argmin(a_dist)
        chosen_mask[closest] = True
        pose_order = [-2, closest]
        dist_path = [a_dist[closest]]
        thetas = [apose, poses[closest]]

        for c in range(pose_num - 1):
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
                        vs, _, _ = smpl(np.tile(np.expand_dims(beta, 0), [len(inter_poses), 1]), inter_poses, batch=True)
                        intersec = False
                        for v in vs:
                            if is_intersec(v, smpl.base.faces):
                                intersec = True
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
        if DEBUG:
            # renderer = Renderer(img_size=448)
            for i, po in enumerate(pose_order):
                if i < 2067:
                    continue
                if po >= 0:
                    v = verts[po]
                else:
                    v, _, _ = smpl(np.zeros([10], np.float32), thetas[i])

                # img = renderer(v, smpl.base.faces)
                pair, _ = get_selfintersections(v, smpl.base.faces)
                # cv2.putText(img, "{} {} {}".format(i, pair.shape[0], po >= 0), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                print("{} {} {}".format(i, pair.shape[0], po >= 0))
                # cv2.imshow('img', img)
                # cv2.waitKey()

        glob_pose_order = find_index(dataset.poses, thetas)
        all_thetas.append(thetas)
        all_pose_orders.append(glob_pose_order)
    all_thetas = np.concatenate(all_thetas, 0)
    all_pose_orders = np.concatenate(all_pose_orders, 0)
    return all_thetas, all_pose_orders


def gen_pose_two_shapes(pose_num, beta1, beta2, nfold=1, div_thresh=0.1, DEBUG=False, gender='neutral',
             intersection=True, garment_class=None, pose_split=None):
    """
    if beta is not None, we will discard frames with self-intersection
    """
    t0 = time.time()
    apose = get_Apose()
    smpl = SMPLNP(gender=gender)

    dataset = Dataset(split=pose_split)
    data_num = len(dataset)
    poses = np.copy(dataset.poses)

    if intersection:
        verts1, _, _ = smpl(np.tile(np.expand_dims(beta1, 0), [poses.shape[0], 1]), poses, batch=True)
        verts2, _, _ = smpl(np.tile(np.expand_dims(beta2, 0), [poses.shape[0], 1]), poses, batch=True)
        good_poses = []
        for v1, v2, p in zip(verts1, verts2, poses):
            if not is_intersec(v1, smpl.base.faces) and not is_intersec(v2, smpl.base.faces):
                good_poses.append(p)
        poses = np.array(good_poses)
        data_num = poses.shape[0]

    if 0 < pose_num < data_num:
        random_idx = np.arange(data_num)
        np.random.shuffle(random_idx)
        data_num = pose_num * nfold
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
        verts, joints, _ = smpl(np.tile(np.expand_dims(beta1, 0), [data_num, 1]), poses, batch=True)

        _, apose_joints, _ = smpl(beta1, apose)

        pose_num = joints.shape[0]
        chosen_mask = np.zeros([joints.shape[0]], dtype=np.bool)
        dist = calc_dis(joints, joints, garment_class)

        try:
            a_dist = calc_dis(np.expand_dims(apose_joints, 0), joints, garment_class)[0]
        except IndexError as e:
            import IPython; IPython.embed()
        closest = np.argmin(a_dist)
        chosen_mask[closest] = True
        pose_order = [-2, closest]
        dist_path = [a_dist[closest]]
        thetas = [apose, poses[closest]]

        for c in range(pose_num - 1):
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
                        vs1, _, _ = smpl(np.tile(np.expand_dims(beta1, 0), [len(inter_poses), 1]), inter_poses, batch=True)
                        vs2, _, _ = smpl(np.tile(np.expand_dims(beta2, 0), [len(inter_poses), 1]), inter_poses, batch=True)
                        intersec = False
                        for v1, v2 in zip(vs1, vs2):
                            if is_intersec(v1, smpl.base.faces) or is_intersec(v2, smpl.base.faces):
                                intersec = True
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
        if DEBUG:
            # renderer = Renderer(img_size=448)
            for i, po in enumerate(pose_order):
                if i < 2067:
                    continue
                if po >= 0:
                    v = verts[po]
                else:
                    v, _, _ = smpl(np.zeros([10], np.float32), thetas[i])

                # img = renderer(v, smpl.base.faces)
                pair, _ = get_selfintersections(v, smpl.base.faces)
                # cv2.putText(img, "{} {} {}".format(i, pair.shape[0], po >= 0), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
                print("{} {} {}".format(i, pair.shape[0], po >= 0))
                # cv2.imshow('img', img)
                # cv2.waitKey()

        glob_pose_order = find_index(dataset.poses_nosplit, thetas)
        all_thetas.append(thetas)
        all_pose_orders.append(glob_pose_order)
    all_thetas = np.concatenate(all_thetas, 0)
    all_pose_orders = np.concatenate(all_pose_orders, 0)
    return all_thetas, all_pose_orders


if __name__ == '__main__':
    div_thresh = 0.15
    DEBUG = False

    # people_name = 'rp_scott_rigged_005_zup_a'
    # people_dir = os.path.join(global_var.DATA_DIR, 'neutral_cloth_test', people_name)
    # label = np.load(os.path.join(people_dir, 'labels.npz'))
    # beta = label['betas']

    thetas, pose_order = gen_pose(50, nfold=1, div_thresh=div_thresh, DEBUG=DEBUG, intersection=True, garment_class='smooth_TShirtNoCoat')
    # if not DEBUG:
    #     np.savez(os.path.join(global_var.DATA_DIR, 'closest_static_pose.npz'),
    #              pose_order=pose_order, thetas=thetas)