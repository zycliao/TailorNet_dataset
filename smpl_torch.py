import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from global_var import ROOT

def batch_rodrigues(theta):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def batch_global_rigid_transformation(Rs, Js, parent, device, rotate_base=False):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = torch.from_numpy(np_rot_x).float().to(device)
        root_rotation = torch.matmul(Rs[:, 0, :, :], rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, torch.ones(N, 1, 1).to(device)], dim=1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim=1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, torch.zeros(N, 24, 1, 1).to(device)], dim=2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A


def batch_lrotmin(theta):
    theta = theta[:, 3:].contiguous()
    Rs = batch_rodrigues(theta.view(-1, 3))
    print(Rs.shape)
    e = torch.eye(3).float()
    Rs = Rs.sub(1.0, e)

    return Rs.view(-1, 23 * 9)


def batch_orth_proj(X, camera):
    '''
        X is N x num_points x 3
    '''
    camera = camera.view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    shape = X_trans.shape
    return (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)


class TorchSMPL4Garment(nn.Module):
    def __init__(self, gender):
        super(TorchSMPL4Garment, self).__init__()

        # with open(model_path, 'rb') as reader:
        #     model = pickle.load(reader, encoding='iso-8859-1')
        model = np.load(os.path.join(ROOT, 'smpl/smpl_hres_{}.npz'.format(gender)))
        with open(os.path.join(ROOT, 'garment_class_info.pkl'), 'rb') as f:
            class_info = pickle.load(f, encoding='latin-1')
        for k in class_info.keys():
            if isinstance(class_info[k]['vert_indices'], np.ndarray):
                class_info[k]['vert_indices'] = torch.tensor(class_info[k]['vert_indices'].astype(np.int64))
            if isinstance(class_info[k]['f'], np.ndarray):
                class_info[k]['f'] = torch.tensor(class_info[k]['f'].astype(np.int64))

        self.class_info = class_info
        self.gender = gender

        self.faces = model['f']

        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.np_v_template = np_v_template

        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)[:, :, :]
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = np.array(model['J_regressor'], dtype=np.float).T
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table']).astype(np.int32)

        np_joint_regressor = np.array(model['J_regressor'], dtype=np.float)
        self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        np_weights = np.array(model['weights'], dtype=np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None
        self.num_verts = 27554

        # skirt_weight: n_skirt, n_body
        skirt_weight = np.load(os.path.join(ROOT, 'skirt_weight.npz'))['w']
        self.register_buffer('skirt_weight', torch.from_numpy(skirt_weight).float())
        skirt_skinning = skirt_weight @ np_weights
        self.register_buffer('skirt_skinning', torch.from_numpy(skirt_skinning[None]).float())

    def save_obj(self, verts, obj_mesh_name):
        if self.faces is None:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

    def forward(self, theta, beta=None, garment_d=None, garment_class=None, rotate_base=False):
        if garment_class == 'skirt':
            return self.forward_skirt(theta, beta, garment_d, rotate_base)
        if not self.cur_device:
            device = theta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = theta.shape[0]

        if beta is not None:
            beta = beta[:, :self.num_betas]
            v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_shaped = self.v_template.unsqueeze(0).expand(num_batch, -1, -1)
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, self.cur_device, rotate_base=rotate_base)

        W = self.weight.view(1, self.num_verts, 24).repeat(num_batch, 1, 1)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
        v_body = v_homo[:, :, :3, 0]

        if garment_class is not None and garment_d is not None:
            vert_indices = self.class_info[garment_class]['vert_indices']
            v_deformed = v_posed.clone()
            v_deformed = v_deformed[:, vert_indices] + garment_d
            v_posed_homo = torch.cat([v_deformed, torch.ones(num_batch, v_deformed.shape[1], 1, device=self.cur_device)], dim=2)
            v_homo = torch.matmul(T[:, vert_indices], torch.unsqueeze(v_posed_homo, -1))
            v_garment = v_homo[:, :, :3, 0]
            return v_body, v_garment
        else:
            return v_body

    def forward_unpose_deformation(self, theta, beta=None, verts=None, garment_class=None, rotate_base=False):
        if garment_class == 'skirt':
            return self.forward_unpose_skirt(theta, beta, verts, rotate_base)
        if not self.cur_device:
            device = theta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = theta.shape[0]

        if beta is not None:
            beta = beta[:, :self.num_betas]
            v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_shaped = self.v_template.unsqueeze(0).expand(num_batch, -1, -1)
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, self.cur_device, rotate_base=rotate_base)

        W = self.weight.view(1, self.num_verts, 24).repeat(num_batch, 1, 1)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        # TODO: this is not working with GPU, find out why
        Tinv = torch.inverse(T)

        indices = self.class_info[garment_class]['vert_indices'].to(self.cur_device)
        if garment_class != None:
            Tinv = Tinv[:,  indices]
            v_posed = v_posed[:, indices]

        verts_homo = torch.cat([verts, torch.ones(num_batch, verts.shape[1], 1, device=self.cur_device)], dim=2)
        v_def = torch.matmul(Tinv, verts_homo.unsqueeze(-1))[:, :, :3, 0] - v_posed
        return v_def

    def forward_skirt(self, theta, beta, garment_d, rotate_base):
        if not self.cur_device:
            device = theta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = theta.shape[0]

        if beta is not None:
            beta = beta[:, :self.num_betas]
            v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_shaped = self.v_template.unsqueeze(0).expand(num_batch, -1, -1)
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, self.cur_device,
                                                                  rotate_base=rotate_base)

        W = self.weight.view(1, self.num_verts, 24).repeat(num_batch, 1, 1)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
        v_body = v_homo[:, :, :3, 0]

        skirt_W = self.skirt_skinning.repeat(num_batch, 1, 1)
        skirt_T = torch.matmul(skirt_W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_deformed = v_posed.clone()
        v_skirt = torch.einsum('sb,nbt->nst', self.skirt_weight, v_deformed)
        v_skirt = v_skirt + garment_d
        v_skirt_homo = torch.cat([v_skirt, torch.ones(num_batch, v_skirt.shape[1], 1, device=self.cur_device)], dim=2)
        v_skirt = torch.matmul(skirt_T, torch.unsqueeze(v_skirt_homo, -1))
        v_skirt = v_skirt[:, :, :3, 0]
        return v_body, v_skirt

    def forward_unpose_skirt(self, theta, beta, verts, rotate_base):
        if not self.cur_device:
            device = theta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = theta.shape[0]

        if beta is not None:
            beta = beta[:, :self.num_betas]
            v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_shaped = self.v_template.unsqueeze(0).expand(num_batch, -1, -1)
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, self.cur_device, rotate_base=rotate_base)

        W = self.weight.view(1, self.num_verts, 24).repeat(num_batch, 1, 1)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        skirt_W = self.skirt_skinning.repeat(num_batch, 1, 1)
        skirt_T = torch.matmul(skirt_W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        skirt_Tinv = torch.inverse(skirt_T)
        v_deformed = v_posed.clone()
        v_skirt = torch.einsum('sb,nbt->nst', self.skirt_weight, v_deformed)

        verts_homo = torch.cat([verts, torch.ones(num_batch, verts.shape[1], 1, device=self.cur_device)], dim=2)
        v_def = torch.matmul(skirt_Tinv, verts_homo.unsqueeze(-1))[:, :, :3, 0] - v_skirt
        return v_def

import sys
class SMPL_Lres(nn.Module):
    def __init__(self, gender):
        super(SMPL_Lres, self).__init__()

        # with open(model_path, 'rb') as reader:
        #     model = pickle.load(reader, encoding='iso-8859-1')
        model = np.load(os.path.join(ROOT, 'smpl/smpl_{}.npz'.format(gender)))
        if sys.version_info.major == 3:
            with open(os.path.join(ROOT, 'garment_class_info.pkl'), 'rb') as f:
                class_info = pickle.load(f, encoding='latin-1')
        else:
            with open(os.path.join(ROOT, 'garment_class_info_py2.pkl'), 'rb') as f:
                class_info = pickle.load(f)
        for k in class_info.keys():
            if isinstance(class_info[k]['vert_indices'], np.ndarray):
                class_info[k]['vert_indices'] = torch.tensor(class_info[k]['vert_indices'].astype(np.int64))
            if isinstance(class_info[k]['f'], np.ndarray):
                class_info[k]['f'] = torch.tensor(class_info[k]['f'].astype(np.int64))

        self.class_info = class_info
        self.gender = gender

        self.faces = model['f'].astype(np.int)

        np_v_template = np.array(model['v_template'], dtype=np.float)
        self.np_v_template = np_v_template

        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]

        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)[:, :, :]
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())

        np_J_regressor = np.array(model['J_regressor'], dtype=np.float).T
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())

        self.parents = np.array(model['kintree_table']).astype(np.int32)

        np_joint_regressor = np.array(model['J_regressor'], dtype=np.float)
        self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())

        np_weights = np.array(model['weights'], dtype=np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())
        self.cur_device = None
        self.num_verts = 6890

    def save_obj(self, verts, obj_mesh_name):
        if self.faces is None:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces:  # Faces are 1-based, not 0-based in obj files
                fp.write('f %d %d %d\n' % (f[0] + 1, f[1] + 1, f[2] + 1))

    def forward(self, theta, beta=None, rotate_base=False):
        if not self.cur_device:
            device = theta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = theta.shape[0]

        if beta is not None:
            beta = beta[:, :self.num_betas]
            v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        else:
            v_shaped = self.v_template.unsqueeze(0).expand(num_batch, -1, -1)
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)

        Rs = batch_rodrigues(theta.contiguous().view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped

        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, self.cur_device, rotate_base=rotate_base)

        W = self.weight.view(1, self.num_verts, 24).repeat(num_batch, 1, 1)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device=self.cur_device)], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))
        v_body = v_homo[:, :, :3, 0]
        return v_body


class SMPLNP(object):
    def __init__(self, gender='neutral', cuda=False):
        self.base = TorchSMPL4Garment(gender)
        if cuda:
            self.cuda = True
            self.base.cuda()
        else:
            self.cuda = False

    def __call__(self, beta, theta, garment_d, garment_class, batch=False):
        if not batch:
            beta = np.expand_dims(beta, 0)
            theta = np.expand_dims(theta, 0)
            if garment_class is not None:
                garment_d = np.expand_dims(garment_d, 0)
        beta = torch.FloatTensor(beta)
        theta = torch.FloatTensor(theta)
        if garment_class is not None:
            garment_d = torch.FloatTensor(garment_d)
        if self.cuda:
            beta = beta.cuda()
            theta = theta.cuda()
            if garment_class is not None:
                garment_d = garment_d.cuda()
        all_vbody, all_vcloth = [], []
        N = beta.shape[0]
        iter_num = int(np.ceil(np.true_divide(N, 100)))
        for i in range(iter_num):
            lp = i*100
            rp = np.minimum((i+1)*100, N)
            if garment_class is None:
                vbody = self.base.forward(theta=theta[lp: rp], beta=beta[lp: rp])
                all_vbody.append(vbody.detach().cpu().numpy())
                all_vcloth.append(vbody.detach().cpu().numpy())
            else:
                vbody, vcloth = self.base.forward(theta=theta[lp: rp], beta=beta[lp: rp], garment_d=garment_d[lp: rp], garment_class=garment_class)
                all_vbody.append(vbody.detach().cpu().numpy())
                all_vcloth.append(vcloth.detach().cpu().numpy())
        all_vbody = np.concatenate(all_vbody, 0)
        all_vcloth = np.concatenate(all_vcloth, 0)
        if not batch:
            all_vbody = all_vbody[0]
            all_vcloth = all_vcloth[0]
        return all_vbody, all_vcloth


class SMPLNP_Lres(object):
    def __init__(self, gender='neutral', cuda=False):
        self.base = SMPL_Lres(gender)
        if cuda:
            self.cuda = True
            self.base.cuda()
        else:
            self.cuda = False

    def __call__(self, beta, theta, batch=False, return_J=False):
        if not batch:
            beta = np.expand_dims(beta, 0)
            theta = np.expand_dims(theta, 0)
        beta = torch.FloatTensor(beta)
        theta = torch.FloatTensor(theta)
        if self.cuda:
            beta = beta.cuda()
            theta = theta.cuda()
        all_vbody, all_Js = [], []
        N = beta.shape[0]
        iter_num = int(np.ceil(np.true_divide(N, 100)))
        for i in range(iter_num):
            lp = i*100
            rp = np.minimum((i+1)*100, N)
            vbody = self.base.forward(theta=theta[lp: rp], beta=beta[lp: rp])
            all_vbody.append(vbody.detach().cpu().numpy())
            all_Js.append(self.base.J_transformed.detach().cpu().numpy())
        all_vbody = np.concatenate(all_vbody, 0)
        all_Js = np.concatenate(all_Js, 0)
        if not batch:
            all_vbody = all_vbody[0]
            all_Js = all_Js[0]
        if return_J:
            return all_vbody, all_Js
        else:
            return all_vbody
