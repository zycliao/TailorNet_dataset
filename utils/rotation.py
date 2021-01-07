import cv2
import os
import pickle
import numpy as np

import global_var


def expmap2rotmat(r):
    """
    :param r: Axis-angle, Nx3
    :return: Rotation matrix, Nx3x3
    """
    EPS = 1e-8
    assert r.shape[1] == 3
    bs = r.shape[0]
    theta = np.sqrt(np.sum(np.square(r), 1, keepdims=True))
    cos_theta = np.expand_dims(np.cos(theta), -1)
    sin_theta = np.expand_dims(np.sin(theta), -1)
    eye = np.tile(np.expand_dims(np.eye(3), 0), (bs, 1, 1))
    norm_r = r / (theta + EPS)
    r_1 = np.expand_dims(norm_r, 2)  # N, 3, 1
    r_2 = np.expand_dims(norm_r, 1)  # N, 1, 3
    zero_col = np.zeros([bs, 1]).astype(r.dtype)
    skew_sym = np.concatenate([zero_col, -norm_r[:, 2:3], norm_r[:, 1:2], norm_r[:, 2:3], zero_col,
                          -norm_r[:, 0:1], -norm_r[:, 1:2], norm_r[:, 0:1], zero_col], 1)
    skew_sym = skew_sym.reshape(bs, 3, 3)
    R = cos_theta*eye + (1-cos_theta)*np.einsum('npq,nqu->npu', r_1, r_2) + sin_theta*skew_sym
    return R


def rotmat2expmap(R):
    """
    :param R: Rotation matrix, Nx3x3
    :return: r: Rotation vector, Nx3
    """
    assert R.shape[1] == R.shape[2] == 3
    theta = np.arccos(np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, -1., 1.)).reshape([-1, 1])
    r = np.stack((R[:, 2, 1]-R[:, 1, 2], R[:, 0, 2]-R[:, 2, 0], R[:, 1, 0]-R[:, 0, 1]), 1) / (2*np.sin(theta))
    r_norm = r / np.sqrt(np.sum(np.square(r), 1, keepdims=True))
    return theta * r_norm


def interpolate_pose(pose1, pose2, inter_num):
    """
    linear interpolation between two axis-angle
    """
    p1 = np.expand_dims(pose1, 0)
    delta = (pose2-pose1) / (inter_num + 1)
    linspace = np.arange(0, inter_num+2)
    for _ in range(pose1.ndim):
        linspace = np.expand_dims(linspace, -1)
    return linspace * delta + p1


def flip_theta(theta, batch=False):
    """
    flip SMPL theta along y-z plane
    if batch is True, theta shape is Nx72, otherwise 72
    """
    exg_idx = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
    if batch:
        new_theta = np.reshape(theta, [-1, 24, 3])
        new_theta = new_theta[:, exg_idx]
        new_theta[:, :, 1:3] *= -1
    else:
        new_theta = np.reshape(theta, [24, 3])
        new_theta = new_theta[exg_idx]
        new_theta[:, 1:3] *= -1
    new_theta = new_theta.reshape(theta.shape)
    return new_theta


def root_rotation(theta):
    """
    rotate along x axis for 180 degree
    """
    rotmat = np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]], dtype=theta.dtype)
    x = cv2.Rodrigues(theta[:3])[0]
    y = cv2.Rodrigues(np.matmul(x, rotmat))[0][:, 0]
    return np.concatenate([y, theta[3:]], 0)


import sys
def get_Apose():
    if sys.version_info.major == 3:
        with open(os.path.join(global_var.ROOT, 'apose.pkl'), 'rb') as f:
            APOSE = np.array(pickle.load(f, encoding='latin-1')['pose']).astype(np.float32)
    else:
        with open(os.path.join(global_var.ROOT, 'apose.pkl'), 'rb') as f:
            APOSE = np.array(pickle.load(f)['pose']).astype(np.float32)
    flip_pose = flip_theta(APOSE)
    APOSE[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15]] = 0
    APOSE[[14, 17, 19, 21, 23]] = flip_pose[[14, 17, 19, 21, 23]]
    APOSE = APOSE.reshape([72])
    return APOSE


def normalize_y_rotation(raw_theta):
    """
    rotate along y axis so that root rotation can always face the camera
    theta should be a [3] or [72] numpy array
    """
    only_global = True
    if raw_theta.shape == (72,):
        theta = raw_theta[:3]
        only_global = False
    else:
        theta = raw_theta[:]
    raw_rot = cv2.Rodrigues(theta)[0]
    rot_z = raw_rot[:, 2]
    # we should rotate along y axis counter-clockwise for t rads to make the object face the camera
    if rot_z[2] == 0:
        t = (rot_z[0] / np.abs(rot_z[0])) * np.pi / 2
    elif rot_z[2] > 0:
        t = np.arctan(rot_z[0]/rot_z[2])
    else:
        t = np.arctan(rot_z[0]/rot_z[2]) + np.pi
    cost, sint = np.cos(t), np.sin(t)
    norm_rot = np.array([[cost, 0, -sint],[0, 1, 0],[sint, 0, cost]])
    final_rot = np.matmul(norm_rot, raw_rot)
    final_theta = cv2.Rodrigues(final_rot)[0][:, 0]
    if not only_global:
        return np.concatenate([final_theta, raw_theta[3:]], 0)
    else:
        return final_theta


def diff_y_rotation(theta1, theta2):
    """
    subtract the y rotation of theta2 by theta1
    return value is an angle which lies in [-pi, pi]
    """
    if theta1.shape == (72,):
        theta1 = theta1[:3]
    if theta2.shape == (72,):
        theta2 = theta2[:3]
    rot_z_1 = cv2.Rodrigues(theta1)[0][:, 2]
    rot_z_2 = cv2.Rodrigues(theta2)[0][:, 2]
    rot_z_1[1] = 0
    rot_z_2[1] = 0
    rot_z_1 = rot_z_1 / np.linalg.norm(rot_z_1)
    rot_z_2 = rot_z_2 / np.linalg.norm(rot_z_2)
    dtheta = np.arccos(np.clip(np.dot(rot_z_1, rot_z_2), a_min=-1, a_max=1))
    if np.cross(rot_z_1, rot_z_2)[1] < 0:
        dtheta *= -1
    return dtheta


def rotate_y(raw_theta, deg):
    """
    rotate along y axis by some degree
    """
    only_global = True
    if raw_theta.shape == (72,):
        theta = raw_theta[:3]
        only_global = False
    else:
        theta = raw_theta[:]
    raw_rot = cv2.Rodrigues(theta)[0]
    cost, sint = np.cos(deg), np.sin(deg)
    y_rot = np.array([[cost, 0, -sint], [0, 1, 0], [sint, 0, cost]])
    final_rot = np.matmul(y_rot, raw_rot)
    final_theta = cv2.Rodrigues(final_rot)[0][:, 0]
    if not only_global:
        return np.concatenate([final_theta, raw_theta[3:]], 0)
    else:
        return final_theta


def angle_y(raw_theta):
    """
    calculate y
    """
    if raw_theta.shape == (72,):
        theta = raw_theta[:3]
    else:
        theta = raw_theta[:]
    raw_rot = cv2.Rodrigues(theta)[0]
    rot_z = raw_rot[:, 2]
    # we should rotate along y axis counter-clockwise for t rads to make the object face the camera
    if rot_z[2] == 0:
        t = (rot_z[0] / np.abs(rot_z[0])) * np.pi / 2
    elif rot_z[2] > 0:
        t = np.arctan(rot_z[0]/rot_z[2])
    else:
        t = np.arctan(rot_z[0]/rot_z[2]) + np.pi
    return t


if __name__ == '__main__':
    apose = get_Apose().astype(np.float32)
    np.save(os.path.join(global_var.ROOT, 'apose.npy'), apose)