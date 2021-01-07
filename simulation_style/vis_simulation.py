import os
import os.path as osp
import cv2
import numpy as np
import trimesh
from utils.renderer import Renderer
from global_var import ROOT


if __name__ == '__main__':
    garment_class = 'skirt'
    gender = 'female'
    renderer = Renderer(512)

    data_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender), 'style_shape')
    shape_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender), 'shape')
    save_dir = osp.join(ROOT, '{}_{}'.format(garment_class, gender), 'style_shape_vis')
    os.makedirs(save_dir, exist_ok=True)

    betas = [k.replace('.obj', '') for k in os.listdir(shape_dir) if k.endswith('.obj')]
    all_vbody = {}
    body_f = None
    for beta in betas:
        m = trimesh.load(osp.join(shape_dir, '{}.obj'.format(beta)), process=False)
        all_vbody[beta] = np.array(m.vertices)
        body_f = np.array(m.faces)

    all_style_shape = [k for k in os.listdir(data_dir) if k.endswith('.obj') and k.startswith('beta')]
    for style_shape in all_style_shape:
        style_shape_path = osp.join(data_dir, style_shape)
        style_shape_save_path = osp.join(save_dir, style_shape)
        beta = style_shape.split('_gamma')[0].replace('beta', '')

        garment_m = trimesh.load(style_shape_path, process=False)
        garment_v = garment_m.vertices
        garment_v[:, 1] -= 2
        img = renderer([all_vbody[beta], garment_v], [body_f, garment_m.faces],
            [np.array([0.6, 0.6, 0.9]), np.array([0.8, 0.5, 0.3])], trans=[1, 0, 0])
        img_back = renderer([all_vbody[beta], garment_v], [body_f, garment_m.faces],
                       [np.array([0.6, 0.6, 0.9]), np.array([0.8, 0.5, 0.3])], trans=[1, 0, 0], euler=[180, 0, 0])
        cv2.imwrite(osp.join(save_dir, style_shape.replace('.obj', '.jpg')), np.concatenate((img, img_back), 1))
