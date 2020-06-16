import os
import os.path as osp
import cv2
import numpy as np
import pickle
from renderer import Renderer
from smpl_torch import SMPLNP
from global_var import DATA_DIR


if __name__ == '__main__':
    garment_class = 't-shirt'
    gender = 'female'
    img_size = 512
    renderer = Renderer(img_size)
    smpl = SMPLNP(gender=gender, cuda=False)

    pose_dir = osp.join(DATA_DIR, '{}_{}'.format(garment_class, gender), 'pose')
    shape_dir = osp.join(DATA_DIR, '{}_{}'.format(garment_class, gender), 'shape')
    ss_dir = osp.join(DATA_DIR, '{}_{}'.format(garment_class, gender), 'style_shape')
    pose_vis_dir = osp.join(DATA_DIR, '{}_{}'.format(garment_class, gender), 'pose_vis')
    ss_vis_dir = osp.join(DATA_DIR, '{}_{}'.format(garment_class, gender), 'style_shape_vis')
    pivots_path = osp.join(DATA_DIR, '{}_{}'.format(garment_class, gender), 'pivots.txt')
    avail_path = osp.join(DATA_DIR, '{}_{}'.format(garment_class, gender), 'avail.txt')
    os.makedirs(pose_vis_dir, exist_ok=True)
    os.makedirs(ss_vis_dir, exist_ok=True)

    with open(os.path.join(DATA_DIR, 'garment_class_info.pkl'), 'rb') as f:
        class_info = pickle.load(f, encoding='latin-1')
    body_f = smpl.base.faces
    garment_f = class_info[garment_class]['f']

    # 1. Visualize pivots data
    with open(pivots_path) as f:
        all_pivots = f.read().strip().splitlines()
    for ss in all_pivots:
        beta_str, gamma_str = ss.split('_')
        pose_ss_dir = osp.join(pose_dir, ss)
        if not osp.exists(pose_ss_dir):
            continue
        unpose_names = [k for k in os.listdir(pose_ss_dir) if k.startswith('unposed') and k.endswith('.npy')]
        beta = np.load(osp.join(shape_dir, 'beta_{}.npy'.format(beta_str)))
        for unpose_name in unpose_names:
            seq_str = unpose_name.replace('unposed_', '').replace('.npy', '')
            pose_path = osp.join(pose_ss_dir, 'poses_{}.npz'.format(seq_str))
            unpose_path = osp.join(pose_ss_dir, unpose_name)
            save_path = osp.join(pose_vis_dir, ss + '_{}.mp4'.format(seq_str))

            unpose_v = np.load(unpose_path)
            thetas = np.load(pose_path)['thetas']
            n = thetas.shape[0]

            all_body, all_gar = smpl(np.tile(beta[None], [n, 1]), thetas, unpose_v, garment_class, batch=True)

            video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), 6., (img_size*2, img_size))

            for i, (body_v, gar_v) in enumerate(zip(all_body, all_gar)):
                img = renderer([body_v, gar_v], [body_f, garment_f],
                    [np.array([0.6, 0.6, 0.9]), np.array([0.8, 0.5, 0.3])], center=True)
                img_back = renderer([body_v, gar_v], [body_f, garment_f],
                    [np.array([0.6, 0.6, 0.9]), np.array([0.8, 0.5, 0.3])], center=True, euler=(180, 0, 0))
                video_writer.write(np.concatenate((img, img_back), 1))
            video_writer.release()
            print("{} written".format(save_path))

    # 2. Visualize all style_shape in canonical pose
    with open(avail_path) as f:
        all_ss = f.read().strip().splitlines()
    apose = np.load(osp.join(DATA_DIR, 'apose.npy'))
    for ss in all_ss:
        beta_str, gamma_str = ss.split('_')
        unpose_path = osp.join(ss_dir, 'beta{}_gamma{}.npy'.format(beta_str, gamma_str))
        if not osp.exists(unpose_path):
            continue

        save_path = osp.join(ss_vis_dir, 'beta{}_gamma{}.jpg'.format(beta_str, gamma_str))
        beta = np.load(osp.join(shape_dir, 'beta_{}.npy'.format(beta_str)))
        unpose_v = np.load(unpose_path)
        body_v, gar_v = smpl(beta, apose, unpose_v, garment_class, batch=False)

        img = renderer([body_v, gar_v], [body_f, garment_f],
                       [np.array([0.6, 0.6, 0.9]), np.array([0.8, 0.5, 0.3])], center=True)
        img_back = renderer([body_v, gar_v], [body_f, garment_f],
                            [np.array([0.6, 0.6, 0.9]), np.array([0.8, 0.5, 0.3])], center=True, euler=(180, 0, 0))
        cv2.imwrite(save_path, np.concatenate((img, img_back), 1))
        print("{} written".format(save_path))

