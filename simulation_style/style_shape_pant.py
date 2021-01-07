"""
simulate different style-shape combinations in A-pose
"""
import os
import os.path as osp
import trimesh
import numpy as np
from utils.ios import save_pc2
from utils.rotation import interpolate_pose
import pickle
from smpl_torch import SMPLNP_Lres, SMPLNP
from utils.rotation import get_Apose
from sklearn.decomposition import PCA
from utils.part_body import part_body_faces
from utils.diffusion_smoothing import DiffusionSmoothing as DS

from global_var import ROOT

APOSE = get_Apose()

def gamma_transform(gamma, coeff_mean, coeff_range):
    return coeff_mean + gamma * coeff_range

def retarget(beta, theta, garment_verts, vert_indices, smpl_hres):
    smpl_hres.pose[:] = theta
    smpl_hres.betas[:] = 0
    zero_body = np.array(smpl_hres.r)

    smpl_hres.betas[:] = beta
    reg_body = np.array(smpl_hres.r)
    retarget_d = (zero_body - reg_body)[vert_indices]
    retarget_verts = garment_verts + retarget_d
    return retarget_verts


def find_perpendicular_foot(x0, x1, x2):
    # x0, x1 define a line. x2 is an arbitary point
    x10 = x1 - x0
    x02 = x0 - x2
    return x0 - (x10) * np.dot(x10, x02) / np.dot(x10, x10)


class GarmentAlign(object):
    def __init__(self, gender):
        model = np.load(os.path.join(ROOT, 'smpl/smpl_hres_{}.npz'.format(gender)))
        model_lres = np.load(os.path.join(ROOT, 'smpl/smpl_{}.npz'.format(gender)))
        with open(os.path.join(ROOT, 'garment_class_info.pkl'), 'rb') as f:
            self.class_info = pickle.load(f, encoding='latin-1')
        self.w = np.array(model['J_regressor'], dtype=np.float).T
        self.w_lres = np.array(model_lres['J_regressor'], dtype=np.float).T


    def align(self, gar_v, body_v, gc):
        vert_indices = self.class_info[gc]['vert_indices']
        gar_smpl = np.zeros([self.w.shape[0], 3])
        gar_smpl[vert_indices] = gar_v
        gar_j = np.einsum('vt,vj->jt', gar_smpl, self.w)
        body_j = np.einsum('vt,vj->jt', body_v, self.w_lres)
        if gc == 't-shirt':
            return np.mean(body_j[[16, 17]] - gar_j[[16, 17]], 0, keepdims=True)
        elif gc == 'shirt':
            left_b = np.load(os.path.join(ROOT, 'shirt_left_boundary.npy'))
            right_b = np.load(os.path.join(ROOT, 'shirt_right_boundary.npy'))
            left_b_center = np.mean(gar_v[left_b], 0)
            right_b_center = np.mean(gar_v[right_b], 0)
            # left_trans = find_perpendicular_foot(body_j[18], body_j[20], left_b_center) - left_b_center
            # right_trans = find_perpendicular_foot(body_j[19], body_j[21], right_b_center) - right_b_center
            # final_trans = ((left_trans + right_trans) / 2.)[None]
            # return final_trans

            import torch
            trans = torch.zeros(3, requires_grad=True)
            lbc_torch = torch.tensor(left_b_center.astype(np.float32), requires_grad=True)
            rbc_torch = torch.tensor(right_b_center.astype(np.float32), requires_grad=True)

            body_j_torch = torch.tensor(body_j.astype(np.float32), requires_grad=True)
            gar_j_torch = torch.tensor(gar_j.astype(np.float32))

            def dist(x0, x1, x2):
                x10 = x1 - x0
                return torch.norm(x2 - x0 - x10 * torch.dot(x2-x0, x10) / torch.dot(x10, x10))

            optim = torch.optim.SGD([trans], lr=1e-3)
            for i in range(50):
                optim.zero_grad()
                loss = dist(body_j_torch[18], body_j_torch[20], lbc_torch+trans) + \
                       dist(body_j_torch[19], body_j_torch[21], rbc_torch+trans)

                loss.backward()
                optim.step()
                # print(f"{i}\t{loss.cpu().detach().numpy()}")

            return trans.detach().numpy()[None]
            return np.mean(body_j[[18, 19]] - gar_j[[18, 19]], 0, keepdims=True)
        elif gc == 'pant':
            return np.mean(body_j[[1, 2]] - gar_j[[1, 2]], 0, keepdims=True)


def polygon_area(v):
    x = v[:, 0].copy()
    y = v[:, 2].copy()
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))



if __name__ == '__main__':
    # STAGE1_FNUM = 5
    # STAGE2_FNUM = 5
    # STAGE3_FNUM = 5
    SM_FNUM = 6
    STABLE_FNUM = 4
    END_FNUM = 5
    lowest = -2
    apose = get_Apose()

    with open(osp.join(ROOT, 'garment_class_info.pkl'), 'rb') as f:
        garment_meta = pickle.load(f)
    for gender in ['female', 'male']:
        smpl = SMPLNP_Lres(gender=gender)
        smpl_hres = SMPLNP(gender=gender)
        gar_align = GarmentAlign(gender)
        num_betas = 10 if gender == 'neutral' else 300
        betas = np.zeros([9, num_betas], dtype=np.float32)
        betas[0, 0] = 2
        betas[1, 0] = -2
        betas[2, 1] = 2
        betas[3, 1] = -2
        betas[4, 0] = 1
        betas[5, 0] = -1
        betas[6, 1] = 1
        betas[7, 1] = -1
        vcanonical = smpl(np.zeros_like(betas[0]), apose)

        for gc in ['short-pant']:
            gc_gender_dir = osp.join(ROOT, '{}_{}'.format(gc, gender))
            shape_dir = osp.join(gc_gender_dir, 'shape')
            save_dir = osp.join(gc_gender_dir, 'style_shape')
            style_dir = osp.join(gc_gender_dir, 'style')
            if not osp.exists(shape_dir):
                os.makedirs(shape_dir)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            if not osp.exists(style_dir):
                os.makedirs(style_dir)
            part_faces = part_body_faces(gc)
            ds = DS(smpl.base.np_v_template, part_faces)

            # gamma
            gammas = []
            if gc in ['t-shirt', 'shirt', 'pant', 'short-pant']:
                gammas.append([0., 0., 0., 0.])
                for x1 in np.arange(-1, 1.01, 0.5):
                    for x2 in np.arange(-1, 1.01, 0.5):
                        if x1 == 0 and x2 == 0:
                            continue
                        gammas.append([x1, x2, 0, 0])
            gammas = np.array(gammas, dtype=np.float32)
            np.save(osp.join(style_dir, 'gammas.npy'), gammas)

            style_model = np.load(osp.join(gc_gender_dir, 'style_model.npz'))
            pca = PCA(n_components=4)
            pca.components_ = style_model['pca_w']
            pca.mean_ = style_model['mean']
            coeff_mean = style_model['coeff_mean']
            coeff_range = style_model['coeff_range']
            # trans = style_model['trans']
            faces = garment_meta[gc]['f']
            vert_indices = garment_meta[gc]['vert_indices']
            # upper_boundary = np.load(osp.join(ROOT, '{}_upper_boundary.npy'.format(gc)))

            up_bnd_inds = np.load(osp.join(ROOT, '{}_upper_boundary.npy'.format(gc)))
            waist_body_inds = vert_indices[up_bnd_inds]
            betas_big2thin = np.zeros([20, num_betas])
            betas_big2thin[:, 1] = np.linspace(-3, 3, 20)
            bodies_big2thin_unsmooth, _ = smpl_hres(betas_big2thin, np.tile(apose[None], (20, 1)), None, None, batch=True)
            bodies_big2thin = []
            for unsmooth_body in bodies_big2thin_unsmooth:
                smooth_body = unsmooth_body
                for smooth_i in range(SM_FNUM):
                    smooth_body = ds.smooth(smooth_body, smoothness=0.1)
                bodies_big2thin.append(smooth_body)
            bodies_big2thin = np.array(bodies_big2thin)

            start_betas = []
            for gamma_i, gamma in enumerate(gammas):
                np.save(osp.join(style_dir, 'gamma_{:03d}.npy'.format(gamma_i)), gamma)
                gamma = gamma_transform(gamma, coeff_mean, coeff_range)
                v = pca.inverse_transform(gamma[None]).reshape([-1, 3])
                v[:, 1] -= lowest
                notrans_m = trimesh.Trimesh(vertices=v, faces=faces, process=False)
                notrans_m.export(osp.join(style_dir, '{:03d}.obj'.format(gamma_i)))


                # for each gamma, find a proper beta that fits it well, so that
                # the waist band can be well pinned in MD
                pant_area = polygon_area(v[up_bnd_inds])
                for body_i, body_v in enumerate(bodies_big2thin):
                    body_area = polygon_area(body_v[waist_body_inds])
                    if gc == 'short-pant' and gender == 'male':
                        if body_area * 1.2 < pant_area:
                            break
                    else:
                        if body_area * 1.1 < pant_area:
                            break
                start_beta = betas_big2thin[body_i].copy()
                start_betas.append(start_beta)

                # initial garment mesh
                # body_v, _ = smpl_hres(, apose, None, None)
                body_v = np.copy(bodies_big2thin[body_i])
                gar_v = np.copy(v)
                trans = np.mean(body_v[waist_body_inds]-gar_v[up_bnd_inds], 0, keepdims=True)
                gar_v += trans
                gar_v[:, 1] -= lowest
                m = trimesh.Trimesh(vertices=gar_v, faces=faces, process=False)
                m.export(osp.join(save_dir, 'input_gamma{:03d}.obj'.format(gamma_i)))

                # initial body mesh
                start_body = smpl(start_beta, apose)
                start_body[:, 1] -= lowest
                for smooth_i in range(SM_FNUM):
                    start_body = ds.smooth(start_body, smoothness=0.1)
                m = trimesh.Trimesh(vertices=start_body, faces=part_faces, process=False)
                m.export(osp.join(save_dir, 'up_gamma{:03d}.obj'.format(gamma_i)))

                # shape
                np.save(osp.join(shape_dir, 'betas.npy'), betas)
                for beta_i, beta in enumerate(betas):
                    np.save(osp.join(shape_dir, 'beta_{:03d}.npy'.format(beta_i)), beta)

                    final_body = smpl(beta, apose)
                    final_body[:, 1] -= lowest
                    trans_body = interpolate_pose(start_body, final_body, SM_FNUM-2)
                    vbody = np.concatenate((np.tile(start_body[None], (STABLE_FNUM, 1, 1)), trans_body,
                                            np.tile(final_body[None], (END_FNUM, 1, 1))), 0)

                    save_pc2(vbody, os.path.join(save_dir, 'motion_beta{:03d}_gamma{:03d}.pc2'.format(beta_i, gamma_i)))
                    # save body
                    vv = vbody[-1]
                    vv[:, 1] += lowest
                    mbody = trimesh.Trimesh(vertices=vv, faces=smpl.base.faces, process=False)
                    mbody.export(osp.join(shape_dir, '{:03d}.obj'.format(beta_i)))

