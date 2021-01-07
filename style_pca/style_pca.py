import os
import pickle
import numpy as np
import skimage.io as sio
from utils.renderer import Renderer
from sklearn.decomposition import PCA
from utils.rotation import get_Apose
from global_var import *

if __name__ == '__main__':
    garment_class = 'skirt'
    n_components = 4
    raw_dir = os.path.join(ROOT, 'raw_data')
    data_dir = os.path.join(ROOT, 'pca')
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(ROOT, 'garment_class_info.pkl'), 'rb') as f:
        class_info = pickle.load(f, encoding='latin-1')
    faces = class_info[garment_class]['f']
    vis_dir = os.path.join(data_dir, 'vis_pca_{}'.format(garment_class))
    os.makedirs(vis_dir, exist_ok=True)
    vs = np.load(os.path.join(raw_dir, '{}_smooth.npy'.format(garment_class)))
    bad_path = os.path.join(raw_dir, '{}_bad.txt'.format(garment_class))
    if os.path.exists(bad_path):
        with open(bad_path) as f:
            bad_idx = f.read().splitlines()
        valid_vs = []
        bad_idx = [int(k) for k in bad_idx]
        for i in range(len(vs)):
            if i not in bad_idx:
                valid_vs.append(vs[i])
        vs = np.array(valid_vs)
    if 'skirt' in garment_class:
        skirt_bnd = np.load(os.path.join(ROOT, 'skirt_upper_boundary.npy'))
        skirt_bnd_loc = np.mean(vs[0, skirt_bnd], 0, keepdims=True)
        vs = vs * 1.15
        all_vs = [vs]
        for _ in range(20):
            random_scale_xz = np.random.uniform(-0.2, 0.2, (len(vs), 1, 1)) + 1
            random_scale_y = np.random.uniform(-0.7, 0.7, (len(vs), 1, 1)) + 1
            random_scale = np.concatenate((random_scale_xz, random_scale_y, random_scale_xz), 2)
            all_vs.append(random_scale*vs)
        vs = np.concatenate(all_vs, 0)
        vs = vs - np.mean(vs[:, skirt_bnd], 1, keepdims=True) + skirt_bnd_loc[None]
    if garment_class == 'short-pant':
        vs = vs * 1.1
    data_num = len(vs)
    print(data_num)
    apose = get_Apose()
    pca = PCA(n_components=n_components)
    pca.fit(vs.reshape([data_num, -1]))
    pca_coeff = pca.transform(vs.reshape([data_num, -1]))
    coeff_mean = np.mean(pca_coeff, 0)
    coeff_std = np.std(pca_coeff, 0)
    print(coeff_mean)
    print(coeff_std)
    np.savez(os.path.join(data_dir, 'style_model_{}.npz'.format(garment_class)),
        pca_w=pca.components_, mean=np.mean(vs.reshape([data_num, -1]), axis=0), coeff_mean=coeff_mean, coeff_std=coeff_std)

    # # sample and visualize
    # r = Renderer(512)
    # for i1, e1 in enumerate(np.arange(-2., 2.01, 0.5)):
    #     for i2, e2 in enumerate(np.arange(-1.5, 1.51, 0.5)):
    #         std_coeff = np.array([e1, e2, 0, 0], dtype=np.float32)
    #         coeff = (std_coeff + coeff_mean) * coeff_std
    #         rec_verts = pca.inverse_transform(coeff[None])[0].reshape([-1, 3])
    #         # _, rec_mesh = smpl(np.zeros([10]), apose, rec_verts, garment_class)
    #         img = r(rec_verts, faces, trans=(2., 0, 0))
    #         sio.imsave(os.path.join(vis_dir, '{}_{}.jpg'.format(i1, i2)), img)