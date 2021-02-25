# find correspondce between the skirt and the SMPL body
import pickle
import os.path as osp
import numpy as np
import cv2
from utils.rotation import get_Apose
from global_var import ROOT
from smpl_torch import SMPLNP, TorchSMPL4Garment
from sklearn.decomposition import PCA
from utils.renderer import Renderer


# load template skirt
style_model = np.load(osp.join(ROOT, 'skirt_female', 'style_model.npz'))
pca = PCA(n_components=4)
pca.components_ = style_model['pca_w']
pca.mean_ = style_model['mean']
skirt_v = pca.inverse_transform(np.zeros([1, 4])).reshape([-1, 3])

# move the skirt to the right position
with open(osp.join(ROOT, 'garment_class_info.pkl'), 'rb') as f:
    garment_meta = pickle.load(f)
skirt_f = garment_meta['skirt']['f']
vert_indices = garment_meta['pant']['vert_indices']
up_bnd_inds = np.load(osp.join(ROOT, 'skirt_upper_boundary.npy'))
pant_up_bnd_inds = np.load(osp.join(ROOT, 'pant_upper_boundary.npy'))
waist_body_inds = vert_indices[pant_up_bnd_inds]

smpl = SMPLNP(gender='female')
apose = get_Apose()
body_v, _ = smpl(np.zeros([300]), apose, None, None)
trans = np.mean(body_v[waist_body_inds], 0, keepdims=True) - np.mean(skirt_v[up_bnd_inds], 0, keepdims=True)
skirt_v = skirt_v + trans

skirt_v[:, 0] -= 0.01

p = 1
K = 100

# find closest vertices
dist = np.sqrt(np.sum(np.square(skirt_v[:, None] - body_v[None]), 2))  # n_skirt, n_body
body_ind = np.argsort(dist, 1)[:, :K]
body_dist = np.sort(dist, 1)[:, :K]
# Inverse distance weighting
w = 1/(body_dist**p)
w = w / np.sum(w, 1, keepdims=True)
n_skirt = len(skirt_v)
n_body = len(body_v)
skirt_weight = np.zeros([n_skirt, n_body], dtype=np.float32)
skirt_weight[np.tile(np.arange(n_skirt)[:, None], (1, K)), body_ind] = w
np.savez_compressed('C:/data/v3/skirt_weight.npz', w=skirt_weight)

exit()


# test
renderer = Renderer(512)
smpl = SMPLNP(gender='female', skirt=True)
smpl_torch = TorchSMPL4Garment('female')

import torch
disp = smpl_torch.forward_unpose_deformation(torch.from_numpy(np.zeros([1, 72])).float(), torch.from_numpy(np.zeros([1, 300])).float(),
                                             torch.from_numpy(skirt_v)[None].float())
disp = disp.detach().cpu().numpy()[0]

for t in np.linspace(0, 1, 20):
    theta = np.zeros([72])
    theta[5] = t
    theta[8] = -t
    body_v, gar_v = smpl(np.zeros([300]), theta, disp, 'skirt')
    img = renderer([body_v, gar_v], [smpl.base.faces, skirt_f],
                    [np.array([0.6, 0.6, 0.9]), np.array([0.8, 0.5, 0.3])], trans=[1, 0, 0])
    cv2.imshow('a', img)
    cv2.waitKey()

# disp = skirt_v - body_v[skirt_ind]
# betas = np.zeros([10, 300])
# betas[:, 1] = np.linspace(-2, 2, 10)
# body_vs, _ = smpl(betas, np.tile(apose[None], (10, 1)), None, None, batch=True)
# gar_vs = body_vs[:, skirt_ind] + disp[None]
#
# for i, (body_v, gar_v) in enumerate(zip(body_vs, gar_vs)):
#     img = renderer([body_v, gar_v], [smpl.base.faces, skirt_f],
#                 [np.array([0.6, 0.6, 0.9]), np.array([0.8, 0.5, 0.3])], trans=[1, 0, 0])
#     cv2.imwrite('a{}.jpg'.format(i), img)
#
