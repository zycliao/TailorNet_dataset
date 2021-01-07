import os
import pickle
import numpy as np
import skimage.io as sio
from smpl_torch import SMPLNP
from utils.renderer import Renderer
from utils.rotation import get_Apose


if __name__ == '__main__':
    garment_class = 'skirt'
    data_dir = 'C:/data/v3/raw_data'
    filenames = ['{}_trans.npy'.format(garment_class), '{}_smooth.npy'.format(garment_class)]
    r = Renderer(512)
    apose = get_Apose()
    with open('C:/data/v3/garment_class_info.pkl', 'rb') as f:
        class_info = pickle.load(f, encoding='latin-1')

    for filename in filenames:
        save_dir = os.path.join(data_dir, filename.replace('.npy', ''))
        all_v = np.load(os.path.join(data_dir, filename))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, v in enumerate(all_v):
            # vbody, vcloth = smpl(np.zeros([300]), apose, disp, garment_class)
            img = r(v, class_info[garment_class]['f'], trans=(2., 0, 0))
            sio.imsave(os.path.join(save_dir, '{}.jpg'.format(i)), img)